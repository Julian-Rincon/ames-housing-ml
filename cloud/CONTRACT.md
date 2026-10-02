# SAVI Cloud — Contrato de integración (fuente de verdad para todos los componentes)

Cualquier cambio a este contrato lo decide el integrador (Opus). Si un componente necesita
algo que no está aquí, documéntalo en tu reporte final en vez de inventarlo.

## Entorno
- Cuenta AWS Academy **Learner Lab** (presupuesto $50; agotarlo DESACTIVA la cuenta).
- Región: `us-east-1`. Cuenta: `006840014780` (nunca hardcodear: obtener con `sts.get_caller_identity`).
- **No se pueden crear roles IAM.** Todo usa el rol preexistente `LabRole`
  (`arn:aws:iam::<account>:role/LabRole`, confía en ec2, lambda, sagemaker) y el instance profile
  `LabInstanceProfile`. Las políticas de menor privilegio se entregan como JSON documentado.
- Restricciones Learner Lab: EC2 sólo nano/micro/small/medium/large, **sólo On-Demand**, EBS ≤100 GB gp2/gp3,
  máx 9 instancias / 32 vCPU. SageMaker sólo medium/large/xlarge. Cuotas: `ml.g4dn.xlarge` training=1 y spot=1.
- **Al terminar cada sesión del lab, las EC2 se suspenden y SE REINICIAN al iniciar el lab de nuevo**
  → el EC2 debe ser idempotente (no re-procesar una corrida ya terminada).

## Nombres de recursos (prefijo `savi`)
| Recurso | Nombre |
|---|---|
| Bucket raw | `savi-raw-<account>` |
| Bucket processed | `savi-processed-<account>` |
| EC2 (tag Name) | `savi-cpu-engine` (t3.medium, AL2023 x86_64 vía SSM `/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64`) |
| Security group | `savi-cpu-engine-sg` (VPC default; SIN reglas de entrada; salida sólo TCP 443 0.0.0.0/0) |
| Lambda 1 | `savi-start-ec2` (python3.12, handler `lambda_start_ec2.handler`) |
| Lambda 2 | `savi-start-sagemaker` (python3.12, handler `lambda_start_sagemaker.handler`) |
| Log group EC2 | `/savi/ec2-cpu-engine` (retención 14 días) |
| Training job | `savi-dqn-<runid-sanitizado>` (≤63 chars, `[a-zA-Z0-9-]`) |

## Layout S3
```
s3://savi-raw-<acct>/
  input/<archivo>.txt|.csv          ← SUBIR AQUÍ dispara el pipeline (Lambda 1: prefix input/)
  reference/fhfa_hpi_ames.csv
  reference/residential-properties-with-detail-2024.xlsx
  reference/residential-sales.xlsx  (opcional)
  code/cpu/utils.py
  code/cpu/savi_cpu_pipeline.py
  code/cpu/requirements-ec2.txt
  code/cpu/run_pipeline.sh

s3://savi-processed-<acct>/
  code/sourcedir.tar.gz             ← savi_gpu_sagemaker.py + utils.py (raíz del tar)
  runs/<run_id>/...                 ← artefactos del EC2 (ver utils.ARTIFACTS)
  runs/<run_id>/_SUCCESS.json       ← ÚLTIMO archivo del EC2 → dispara Lambda 2 (prefix runs/, suffix _SUCCESS.json)
  runs/<run_id>/logs/ec2_pipeline.log
  runs/<run_id>/checkpoints/        ← checkpoints Spot de SageMaker
  runs/<run_id>/sagemaker/<job>/output/model.tar.gz
```
`_SUCCESS.json` = `{"run_id": str, "status": "SUCCEEDED", "output": "s3://.../runs/<id>/", "finished_utc": str, "artifacts": {...}}`

## run_id
Lo genera la Lambda 1: `<UTC %Y%m%dT%H%M%SZ>-<stem del archivo sanitizado a [a-z0-9-], máx 20 chars>`.
Ej: `20261002T210501Z-ameshousing`.

## Lambda 1 → EC2 (vía tags de la instancia)
Lambda 1 escribe en la instancia los tags:
- `SaviInputKey` = `s3://savi-raw-<acct>/input/<archivo>`
- `SaviRunId`    = run_id
El EC2 los lee por IMDSv2 (`InstanceMetadataTags=enabled`):
`http://169.254.169.254/latest/meta-data/tags/instance/SaviRunId`.
Tag opcional `SaviNoShutdown=true` → el EC2 NO se apaga al terminar (sólo depuración).

Env vars Lambda 1: `INSTANCE_ID`, `RAW_BUCKET`.
Estados: `stopped`→start; `stopping`→esperar `instance_stopped` y start; `pending|running`→sólo tags
(el script del EC2 re-chequea el tag antes de apagarse). Ignorar keys fuera de `input/` o sin sufijo `.csv/.txt`.
Las keys del evento S3 vienen URL-encoded → `urllib.parse.unquote_plus`.

## EC2: comando del pipeline
```
/opt/savi/venv/bin/python /opt/savi/code/savi_cpu_pipeline.py \
  --input s3://savi-raw-<acct>/input/<archivo> \
  --reference s3://savi-raw-<acct>/reference/ \
  --output s3://savi-processed-<acct>/runs/<run_id>/ \
  --run-id <run_id>
```
Idempotencia: si existe `s3://savi-processed-<acct>/runs/<run_id>/_SUCCESS.json` → no procesar.
Siempre `shutdown -h now` al final (éxito, error o timeout duro de 60 min), salvo `SaviNoShutdown=true`.
`InstanceInitiatedShutdownBehavior=stop` (NO terminate).

## Lambda 2 → SageMaker
Env vars: `SAGEMAKER_ROLE_ARN`, `IMAGE_URI`
(`763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-sagemaker`),
`CPU_IMAGE_URI` (`...pytorch-training:2.7.1-cpu-py312-ubuntu22.04-sagemaker`),
`INSTANCE_TYPE`=`ml.g4dn.xlarge`, `FALLBACK_INSTANCE_TYPE`=`ml.m5.xlarge`, `USE_SPOT`=`true`,
`CODE_S3_URI`=`s3://savi-processed-<acct>/code/sourcedir.tar.gz`, `PROCESSED_BUCKET`, `MAX_RUNTIME`=`3600`, `EPOCHS`=`150`.

`create_training_job`:
- Canal `processed` = `s3://savi-processed-<acct>/runs/<run_id>/` (S3Prefix, FullyReplicated, File mode)
- Output `s3://savi-processed-<acct>/runs/<run_id>/sagemaker/`
- Hiperparámetros (TODOS los valores `json.dumps(...)` como hace el SDK): `sagemaker_program`="savi_gpu_sagemaker.py",
  `sagemaker_submit_directory`=CODE_S3_URI, `sagemaker_region`, `sagemaker_container_log_level`=20,
  `epochs`, `run-id`.
- Spot: `EnableManagedSpotTraining`, `MaxWaitTimeInSeconds`=2×MAX_RUNTIME,
  `CheckpointConfig`={S3Uri: runs/<run_id>/checkpoints/, LocalPath: /opt/ml/checkpoints}.
- Fallback: si falla por cuota/capacidad/permiso (`ResourceLimitExceeded`, `CapacityError`, `AccessDeniedException`
  — Learner Lab niega ml.g4dn.* por política —, o `ValidationException` relacionada con spot)
  → GPU on-demand → `FALLBACK_INSTANCE_TYPE` + `CPU_IMAGE_URI` spot → mismo on-demand.
- Idempotencia: `ResourceInUse` (job ya existe) → no es error.
- `VolumeSizeInGB`=10, `InstanceCount`=1, tags `Project=SAVI`, `RunId`.

## Script GPU (ya escrito: `savi_gpu_sagemaker.py`)
Lee `SM_CHANNEL_PROCESSED`, escribe `SM_MODEL_DIR`, checkpoints en `/opt/ml/checkpoints`.
Acepta `--epochs --batch-size --lr --gamma --target-update --buffer --eps-decay --act-chunk --ckpt-every --run-id`.

## Logging
Todo Python con `logging` nivel INFO a stdout (CloudWatch). Lambdas: logs JSON-friendly con el run_id.
