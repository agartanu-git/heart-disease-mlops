{{- define "heart-disease-api.name" -}}
heart-api
{{- end -}}

{{- define "heart-disease-api.labels" -}}
app: {{ include "heart-disease-api.name" . }}
app.kubernetes.io/name: {{ include "heart-disease-api.name" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}
