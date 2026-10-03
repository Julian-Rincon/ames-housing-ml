# IAM — SAVI Cloud

Estas políticas documentan el acceso de **mínimo privilegio** que necesitaría
cada componente (EC2, Lambda 1, Lambda 2, rol de ejecución de SageMaker) si se
desplegara en una cuenta AWS normal.

## Por qué no se usan en el despliegue por defecto

El entorno real es **AWS Academy Learner Lab**, donde la política del laboratorio
**deniega `iam:CreateRole`, `iam:CreatePolicy` y operaciones similares** para
cualquier usuario del lab. Por eso `infra/deploy.py` usa por defecto el rol
preexistente `LabRole` (confía en `ec2.amazonaws.com`, `lambda.amazonaws.com`
y `sagemaker.amazonaws.com`) y el instance profile `LabInstanceProfile`,
ambos provistos por el lab con permisos amplios (`AdministratorAccess` o
similar) que ya cubren de sobra lo que estas políticas piden.

`deploy.py --create-roles` existe para el caso en que esto se despliegue
**fuera** de Learner Lab (cuenta propia con permisos de IAM reales). En ese
caso se usarían estos JSON como `AssumeRolePolicyDocument` (los `*_trust_policy.json`)
y como política inline/administrada (los `*_role_policy.json`), reemplazando
los placeholders:

- `${ACCOUNT_ID}` — id de cuenta AWS.
- `${REGION}` — región (`us-east-1`).
- `${RAW_BUCKET}` / `${PROCESSED_BUCKET}` — `savi-raw-<acct>` / `savi-processed-<acct>`.
- `${INSTANCE_ID}` — id de la instancia `savi-cpu-engine` (sólo se conoce tras crearla;
  en un primer despliegue habría que usar `Resource: "*"` con condición de tag, o
  crear la política después de la instancia).
- `${SAGEMAKER_ROLE_ARN}` — ARN del rol de ejecución de SageMaker.

## Alcance de cada política

| Archivo | Rol | Permisos |
|---|---|---|
| `ec2_trust_policy.json` / `ec2_role_policy.json` | Instancia `savi-cpu-engine` | Leer `input/`, `reference/`, `code/cpu/` del bucket raw; escribir `runs/*` en processed; leer `_SUCCESS.json` para chequear idempotencia; logs del log group `/savi/ec2-cpu-engine`. |
| `lambda1_trust_policy.json` / `lambda1_role_policy.json` | `savi-start-ec2` | `ec2:StartInstances`/`CreateTags` escopados al ARN de la instancia; `ec2:DescribeInstances` (no es escopable a un recurso); logs propios. |
| `lambda2_trust_policy.json` / `lambda2_role_policy.json` | `savi-start-sagemaker` | Leer `_SUCCESS.json` de processed; `sagemaker:CreateTrainingJob`/`AddTags`; `iam:PassRole` sobre el rol de ejecución de SageMaker con condición `iam:PassedToService=sagemaker.amazonaws.com`; logs propios. |
| `sagemaker_trust_policy.json` / `sagemaker_role_policy.json` | Rol de ejecución del training job | Leer `runs/*` y `code/*` de processed; escribir `runs/*/sagemaker/*` y `runs/*/checkpoints/*`; pull de la imagen DLC de ECR; logs de SageMaker Training. |

No se pide `ec2:DescribeTags` en el rol de EC2 porque el script del motor CPU
lee los tags de la instancia vía IMDSv2 (`InstanceMetadataTags=enabled`), no
vía API de EC2.
