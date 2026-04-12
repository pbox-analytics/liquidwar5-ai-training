/*
 * GPU-native Liquid War simulator.
 *
 * One thread block per game. Entire game state lives in shared memory.
 * All ticks run inside the kernel — no return to Python until game ends.
 *
 * Grid: 64x64, 4 teams, 256 threads per block.
 * Each thread handles 16 cells (4096/256).
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define GRID_H 64
#define GRID_W 64
#define GRID_SIZE (GRID_H * GRID_W)  // 4096
#define MAX_TEAMS 4
#define THREADS_PER_BLOCK 256
#define CELLS_PER_THREAD (GRID_SIZE / THREADS_PER_BLOCK)  // 16
#define MAX_HEALTH 16384
#define GRAD_INF 30000
#define ATTACK 30
#define DEFENSE 10

// Shared memory layout per game
struct GameState {
    int8_t team[GRID_SIZE];           // -1=empty, 0-3=team
    int16_t health[GRID_SIZE];        // 0=no fighter
    int16_t gradient[MAX_TEAMS][GRID_SIZE];  // distance from cursor
    int16_t cursor_y[MAX_TEAMS];
    int16_t cursor_x[MAX_TEAMS];
    int team_alive[MAX_TEAMS];        // 1=alive, 0=eliminated
    int team_fighters[MAX_TEAMS];     // count per team
};

// 8-direction neighbor offsets
__constant__ int DY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
__constant__ int DX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

// Check if position is valid (within grid and not a wall)
__device__ inline bool is_valid(int y, int x, const bool* walls, int game_id) {
    if (y < 0 || y >= GRID_H || x < 0 || x >= GRID_W) return false;
    return !walls[game_id * GRID_SIZE + y * GRID_W + x];
}

// Initialize game state in shared memory
__device__ void init_game(
    GameState* gs,
    const bool* walls,
    int game_id,
    int num_teams,
    int fighters_per_team,
    int tid
) {
    // Initialize all cells handled by this thread
    for (int c = 0; c < CELLS_PER_THREAD; c++) {
        int idx = tid * CELLS_PER_THREAD + c;
        gs->team[idx] = -1;
        gs->health[idx] = 0;
        for (int t = 0; t < MAX_TEAMS; t++)
            gs->gradient[t][idx] = GRAD_INF;
    }

    // Thread 0 initializes cursors and team state
    if (tid == 0) {
        int strip_w = (GRID_W - 2) / num_teams;
        for (int t = 0; t < MAX_TEAMS; t++) {
            if (t < num_teams) {
                gs->cursor_y[t] = GRID_H / 2;
                gs->cursor_x[t] = 1 + t * strip_w + strip_w / 2;
                gs->team_alive[t] = 1;
                gs->team_fighters[t] = 0;
            } else {
                gs->cursor_y[t] = 0;
                gs->cursor_x[t] = 0;
                gs->team_alive[t] = 0;
                gs->team_fighters[t] = 0;
            }
        }
    }
    __syncthreads();

    // Place fighters in vertical strips — simple fill
    int strip_w = (GRID_W - 2) / num_teams;
    for (int c = 0; c < CELLS_PER_THREAD; c++) {
        int idx = tid * CELLS_PER_THREAD + c;
        int y = idx / GRID_W;
        int x = idx % GRID_W;

        if (y < 2 || y >= GRID_H - 2) continue;
        if (walls[game_id * GRID_SIZE + idx]) continue;

        for (int t = 0; t < num_teams; t++) {
            int x_start = 1 + t * strip_w;
            int x_end = x_start + strip_w;
            if (x >= x_start && x < x_end) {
                gs->team[idx] = t;
                gs->health[idx] = MAX_HEALTH;
                break;
            }
        }
    }
    __syncthreads();

    // Count fighters
    if (tid < MAX_TEAMS) gs->team_fighters[tid] = 0;
    __syncthreads();
    for (int c = 0; c < CELLS_PER_THREAD; c++) {
        int idx = tid * CELLS_PER_THREAD + c;
        if (gs->team[idx] >= 0)
            atomicAdd(&gs->team_fighters[(int)gs->team[idx]], 1);
    }
    __syncthreads();
}

// Seed gradient at cursor + age gradient
__device__ void seed_and_age_gradient(GameState* gs, const bool* walls,
                                       int game_id, int num_teams, int tid) {
    for (int c = 0; c < CELLS_PER_THREAD; c++) {
        int idx = tid * CELLS_PER_THREAD + c;
        bool is_wall = walls[game_id * GRID_SIZE + idx];
        for (int t = 0; t < num_teams; t++) {
            if (is_wall) {
                gs->gradient[t][idx] = GRAD_INF;
            } else {
                int16_t g = gs->gradient[t][idx];
                gs->gradient[t][idx] = (g < GRAD_INF - 1) ? g + 1 : GRAD_INF;
            }
        }
    }

    // Seed cursors at 0
    if (tid == 0) {
        for (int t = 0; t < num_teams; t++) {
            if (gs->team_alive[t]) {
                int idx = gs->cursor_y[t] * GRID_W + gs->cursor_x[t];
                gs->gradient[t][idx] = 0;
            }
        }
    }
    __syncthreads();
}

// Spread gradient: min(neighbor) + 1 for all cells
__device__ void spread_gradient(GameState* gs, const bool* walls,
                                 int game_id, int num_teams, int tid,
                                 int iterations) {
    for (int iter = 0; iter < iterations; iter++) {
        for (int c = 0; c < CELLS_PER_THREAD; c++) {
            int idx = tid * CELLS_PER_THREAD + c;
            int y = idx / GRID_W;
            int x = idx % GRID_W;

            if (walls[game_id * GRID_SIZE + idx]) continue;

            for (int t = 0; t < num_teams; t++) {
                int16_t best = gs->gradient[t][idx];
                for (int d = 0; d < 8; d++) {
                    int ny = y + DY[d];
                    int nx = x + DX[d];
                    if (ny >= 0 && ny < GRID_H && nx >= 0 && nx < GRID_W) {
                        int nidx = ny * GRID_W + nx;
                        if (!walls[game_id * GRID_SIZE + nidx]) {
                            int16_t ng = gs->gradient[t][nidx] + 1;
                            if (ng < best) best = ng;
                        }
                    }
                }
                gs->gradient[t][idx] = best;
            }
        }
        __syncthreads();
    }
}

// Simple AI: move cursor toward enemy centroid
__device__ void ai_move_cursors(GameState* gs, const bool* walls,
                                 int game_id, int num_teams, int tid) {
    // Thread 0 computes enemy centroids and moves cursors
    if (tid == 0) {
        for (int t = 0; t < num_teams; t++) {
            if (!gs->team_alive[t]) continue;

            // Compute enemy centroid
            float ey_sum = 0, ex_sum = 0;
            int e_count = 0;
            for (int idx = 0; idx < GRID_SIZE; idx++) {
                if (gs->team[idx] >= 0 && gs->team[idx] != t) {
                    ey_sum += idx / GRID_W;
                    ex_sum += idx % GRID_W;
                    e_count++;
                }
            }
            if (e_count == 0) continue;

            float ey = ey_sum / e_count;
            float ex = ex_sum / e_count;

            int dy = (ey > gs->cursor_y[t]) ? 1 : ((ey < gs->cursor_y[t]) ? -1 : 0);
            int dx = (ex > gs->cursor_x[t]) ? 1 : ((ex < gs->cursor_x[t]) ? -1 : 0);

            int ny = gs->cursor_y[t] + dy;
            int nx = gs->cursor_x[t] + dx;

            if (ny >= 1 && ny < GRID_H - 1 && nx >= 1 && nx < GRID_W - 1 &&
                !walls[game_id * GRID_SIZE + ny * GRID_W + nx]) {
                gs->cursor_y[t] = ny;
                gs->cursor_x[t] = nx;
            }
        }
    }
    __syncthreads();
}

// Move fighters along gradient
__device__ void move_fighters(GameState* gs, const bool* walls,
                               int game_id, int num_teams, int tid,
                               int tick) {
    // Alternate even/odd cells to avoid conflicts
    int parity = tick & 1;

    for (int c = 0; c < CELLS_PER_THREAD; c++) {
        int idx = tid * CELLS_PER_THREAD + c;
        int y = idx / GRID_W;
        int x = idx % GRID_W;

        // Only move cells matching parity (checkerboard pattern)
        if (((y + x) & 1) != parity) continue;
        if (gs->team[idx] < 0) continue;

        int t = gs->team[idx];
        int16_t cur_grad = gs->gradient[t][idx];
        int best_dir = -1;
        int16_t best_grad = cur_grad;

        for (int d = 0; d < 8; d++) {
            int ny = y + DY[d];
            int nx = x + DX[d];
            if (ny < 0 || ny >= GRID_H || nx < 0 || nx >= GRID_W) continue;
            int nidx = ny * GRID_W + nx;
            if (walls[game_id * GRID_SIZE + nidx]) continue;
            if (gs->team[nidx] >= 0) continue;  // occupied

            int16_t ng = gs->gradient[t][nidx];
            if (ng < best_grad) {
                best_grad = ng;
                best_dir = d;
            }
        }

        if (best_dir >= 0) {
            int ny = y + DY[best_dir];
            int nx = x + DX[best_dir];
            int nidx = ny * GRID_W + nx;

            // Try to claim destination — simple check (not perfectly atomic
            // for int8 but races just mean a missed move, not corruption)
            if (gs->team[nidx] == -1) {
                gs->team[nidx] = gs->team[idx];
                gs->health[nidx] = gs->health[idx];
                gs->team[idx] = -1;
                gs->health[idx] = 0;
            }
        }
    }
    __syncthreads();
}

// Combat: damage and capture
__device__ void resolve_combat(GameState* gs, const bool* walls,
                                int game_id, int num_teams, int tid) {
    for (int c = 0; c < CELLS_PER_THREAD; c++) {
        int idx = tid * CELLS_PER_THREAD + c;
        int y = idx / GRID_W;
        int x = idx % GRID_W;

        if (gs->team[idx] < 0) continue;

        int t = gs->team[idx];
        int friendly = 0, enemy = 0;

        for (int d = 0; d < 8; d++) {
            int ny = y + DY[d];
            int nx = x + DX[d];
            if (ny < 0 || ny >= GRID_H || nx < 0 || nx >= GRID_W) continue;
            int nidx = ny * GRID_W + nx;
            if (gs->team[nidx] < 0) continue;
            if (gs->team[nidx] == t) friendly++;
            else enemy++;
        }

        if (enemy > 0) {
            int damage = enemy * ATTACK - friendly * DEFENSE;
            if (damage > 0) {
                gs->health[idx] -= damage;
                if (gs->health[idx] <= 0) {
                    // Find dominant enemy neighbor
                    int team_count[MAX_TEAMS] = {0};
                    for (int d = 0; d < 8; d++) {
                        int ny = y + DY[d];
                        int nx = x + DX[d];
                        if (ny < 0 || ny >= GRID_H || nx < 0 || nx >= GRID_W) continue;
                        int nidx = ny * GRID_W + nx;
                        if (gs->team[nidx] >= 0 && gs->team[nidx] != t)
                            team_count[(int)gs->team[nidx]]++;
                    }

                    int best_team = -1, best_count = 0;
                    for (int tt = 0; tt < num_teams; tt++) {
                        if (team_count[tt] > best_count) {
                            best_count = team_count[tt];
                            best_team = tt;
                        }
                    }

                    if (best_team >= 0) {
                        gs->team[idx] = best_team;
                        gs->health[idx] = MAX_HEALTH / 2;
                    } else {
                        gs->team[idx] = -1;
                        gs->health[idx] = 0;
                    }
                }
            }
        }
    }
    __syncthreads();
}

// Count fighters per team
__device__ void count_fighters(GameState* gs, int num_teams, int tid) {
    // Reset counts
    if (tid < MAX_TEAMS) gs->team_fighters[tid] = 0;
    __syncthreads();

    for (int c = 0; c < CELLS_PER_THREAD; c++) {
        int idx = tid * CELLS_PER_THREAD + c;
        if (gs->team[idx] >= 0)
            atomicAdd(&gs->team_fighters[(int)gs->team[idx]], 1);
    }
    __syncthreads();

    // Check eliminations
    if (tid < MAX_TEAMS) {
        gs->team_alive[tid] = (gs->team_fighters[tid] > 0) ? 1 : 0;
    }
    __syncthreads();
}

// ==================================================================
// Main kernel: one thread block = one complete game
// ==================================================================

__global__ void liquid_war_kernel(
    const bool* __restrict__ walls,     // (B, GRID_H, GRID_W)
    float* __restrict__ results,        // (B, MAX_TEAMS) fighter counts
    int* __restrict__ ticks_out,        // (B,) ticks elapsed
    int B,
    int num_teams,
    int fighters_per_team,
    int max_ticks,
    int grad_iters
) {
    int game_id = blockIdx.x;
    if (game_id >= B) return;
    int tid = threadIdx.x;

    // Shared memory for game state
    __shared__ GameState gs;

    // Initialize
    init_game(&gs, walls, game_id, num_teams, fighters_per_team, tid);

    // Run game
    for (int tick = 0; tick < max_ticks; tick++) {
        // Check if game is over (only after first count at tick 99)
        if (tick > 0 && tick % 100 == 0) {
            int alive_count = 0;
            for (int t = 0; t < num_teams; t++)
                alive_count += gs.team_alive[t];
            if (alive_count <= 1) {
                if (tid == 0) ticks_out[game_id] = tick;
                break;
            }
        }

        // AI
        ai_move_cursors(&gs, walls, game_id, num_teams, tid);

        // Gradient
        seed_and_age_gradient(&gs, walls, game_id, num_teams, tid);
        int iters = (tick < 30) ? 20 : grad_iters;
        spread_gradient(&gs, walls, game_id, num_teams, tid, iters);

        // Movement
        move_fighters(&gs, walls, game_id, num_teams, tid, tick);

        // Combat
        resolve_combat(&gs, walls, game_id, num_teams, tid);

        // Count fighters every 100 ticks for elimination check
        if (tick % 100 == 99)
            count_fighters(&gs, num_teams, tid);

        if (tid == 0 && tick == max_ticks - 1)
            ticks_out[game_id] = max_ticks;
    }

    // Final count
    count_fighters(&gs, num_teams, tid);

    // Write results to global memory
    if (tid < MAX_TEAMS) {
        results[game_id * MAX_TEAMS + tid] = gs.team_fighters[tid];
    }
}

// ==================================================================
// Python binding
// ==================================================================

torch::Tensor run_games(
    torch::Tensor walls,    // (B, 64, 64) bool
    int num_teams,
    int fighters_per_team,
    int max_ticks,
    int grad_iters
) {
    int B = walls.size(0);
    TORCH_CHECK(walls.size(1) == GRID_H && walls.size(2) == GRID_W,
                "walls must be (B, 64, 64)");

    auto results = torch::zeros({B, MAX_TEAMS}, walls.options().dtype(torch::kFloat32));
    auto ticks = torch::zeros({B}, walls.options().dtype(torch::kInt32));

    // One block per game, 256 threads per block
    int grid = B;
    int block = THREADS_PER_BLOCK;

    // Request extended shared memory
    size_t shared_mem = sizeof(GameState);
    cudaFuncSetAttribute(liquid_war_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_mem);

    // Launch kernel
    size_t shared_needed = sizeof(GameState);
    liquid_war_kernel<<<grid, block, shared_needed>>>(
        walls.data_ptr<bool>(),
        results.data_ptr<float>(),
        ticks.data_ptr<int>(),
        B, num_teams, fighters_per_team, max_ticks, grad_iters
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel execution failed: ", cudaGetErrorString(err));

    return torch::cat({results, ticks.unsqueeze(1).to(torch::kFloat32)}, 1);
    // Returns (B, MAX_TEAMS + 1): [team0_fighters, ..., team3_fighters, ticks]
}

// Binding is done by load_inline in gpu_native.py
