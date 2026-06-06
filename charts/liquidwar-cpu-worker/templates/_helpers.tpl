{{/*
Fully-qualified app name. Defaults to "liquidwar-cpu-worker"; per-node
releases override via nameOverride so multiple workers can coexist.
*/}}
{{- define "lwcpu.fullname" -}}
{{- default "liquidwar-cpu-worker" .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}
