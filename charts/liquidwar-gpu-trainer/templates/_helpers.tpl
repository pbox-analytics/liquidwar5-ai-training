{{/*
Fully-qualified app name. Defaults to "liquidwar-gpu-trainer"; overridable
via nameOverride.
*/}}
{{- define "lwgpu.fullname" -}}
{{- default "liquidwar-gpu-trainer" .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}
