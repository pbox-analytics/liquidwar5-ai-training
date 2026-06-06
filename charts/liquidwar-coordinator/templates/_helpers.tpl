{{/*
Fully-qualified app name. Defaults to "liquidwar-coordinator"; overridable
via nameOverride.
*/}}
{{- define "lwcoord.fullname" -}}
{{- default "liquidwar-coordinator" .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}
