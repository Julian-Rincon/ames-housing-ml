# -*- coding: utf-8 -*-
"""
Tests de los Lambdas de SAVI Cloud (lambda_start_ec2 y lambda_start_sagemaker).

Reglas duras: NO moto, NO red, NO credenciales reales de AWS.
Usamos botocore.stub.Stubber para simular las respuestas de EC2/S3/SageMaker.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lambdas"))

import lambda_start_ec2 as L1
import lambda_start_sagemaker as L2


# ════════════════════════════════════════════════════════════════════
# Fixtures comunes
# ════════════════════════════════════════════════════════════════════
@pytest.fixture(autouse=True)
def fake_aws_env(monkeypatch):
    """Credenciales/region falsas: nunca debe tocar AWS real ni ~/.aws."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")


def _s3_event(bucket: str, key: str) -> dict:
    return {"Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]}


# ════════════════════════════════════════════════════════════════════
# Lambda 1 — lambda_start_ec2
# ════════════════════════════════════════════════════════════════════
class TestKeyFiltering:
    def test_ignores_key_outside_input_prefix(self):
        assert L1._is_relevant_key("reference/foo.csv") is False

    def test_ignores_non_csv_txt_suffix(self):
        assert L1._is_relevant_key("input/foo.xlsx") is False

    def test_accepts_csv_case_insensitive(self):
        assert L1._is_relevant_key("input/Ames.CSV") is True

    def test_accepts_txt(self):
        assert L1._is_relevant_key("input/data.txt") is True

    def test_ignores_bare_prefix(self):
        assert L1._is_relevant_key("input/") is False


class TestRunId:
    def test_url_encoded_key_is_unquoted(self):
        # %20 -> espacio, +  -> espacio (unquote_plus)
        raw = "input/ames+housing.csv"
        from urllib.parse import unquote_plus
        assert unquote_plus(raw) == "input/ames housing.csv"

    def test_run_id_format(self):
        now = datetime(2026, 10, 2, 21, 5, 1, tzinfo=timezone.utc)
        run_id = L1.build_run_id("input/AmesHousing.csv", now=now)
        assert run_id == "20261002T210501Z-ameshousing"

    def test_run_id_sanitizes_and_truncates_stem(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        run_id = L1.build_run_id("input/Some Weird File!!__Name_With_Many_Chars.csv", now=now)
        prefix, stem = run_id.split("-", 1)
        assert prefix == "20260101T000000Z"
        assert len(stem) <= 20
        assert all(c.islower() or c.isdigit() or c == "-" for c in stem)

    def test_run_id_matches_example_in_contract(self):
        now = datetime(2026, 10, 2, 21, 5, 1, tzinfo=timezone.utc)
        assert L1.build_run_id("input/ameshousing.csv", now=now) == "20261002T210501Z-ameshousing"


class TestLambda1Handler:
    def _env(self, monkeypatch):
        monkeypatch.setenv("INSTANCE_ID", "i-0123456789abcdef0")
        monkeypatch.setenv("RAW_BUCKET", "savi-raw-006840014780")

    def test_stopped_instance_is_started(self, monkeypatch):
        self._env(monkeypatch)
        ec2 = boto3.client("ec2", region_name="us-east-1")
        stubber = Stubber(ec2)
        monkeypatch.setattr(L1, "_ec2_client", lambda: ec2)

        stubber.add_response("create_tags", {}, {
            "Resources": ["i-0123456789abcdef0"],
            "Tags": ANY,
        })
        stubber.add_response("describe_instances", {
            "Reservations": [{"Instances": [{"State": {"Name": "stopped"}}]}]
        }, {"InstanceIds": ["i-0123456789abcdef0"]})
        stubber.add_response("start_instances", {
            "StartingInstances": [{"InstanceId": "i-0123456789abcdef0",
                                    "CurrentState": {"Name": "pending"},
                                    "PreviousState": {"Name": "stopped"}}]
        }, {"InstanceIds": ["i-0123456789abcdef0"]})

        with stubber:
            result = L1.handler(_s3_event("savi-raw-006840014780", "input/ameshousing.csv"), None)

        assert result["processed"] == 1
        assert result["results"][0]["action"] == "started"
        assert result["results"][0]["instance_state"] == "stopped"
        stubber.assert_no_pending_responses()

    def test_stopping_instance_waits_then_starts(self, monkeypatch):
        self._env(monkeypatch)
        ec2 = boto3.client("ec2", region_name="us-east-1")
        stubber = Stubber(ec2)
        monkeypatch.setattr(L1, "_ec2_client", lambda: ec2)

        stubber.add_response("create_tags", {})
        stubber.add_response("describe_instances", {
            "Reservations": [{"Instances": [{"State": {"Name": "stopping"}}]}]
        }, {"InstanceIds": ["i-0123456789abcdef0"]})
        # Waiter: primer poll todavía 'stopping', segundo ya 'stopped'.
        stubber.add_response("describe_instances", {
            "Reservations": [{"Instances": [{"State": {"Name": "stopping"}}]}]
        }, {"InstanceIds": ["i-0123456789abcdef0"]})
        stubber.add_response("describe_instances", {
            "Reservations": [{"Instances": [{"State": {"Name": "stopped"}}]}]
        }, {"InstanceIds": ["i-0123456789abcdef0"]})
        stubber.add_response("start_instances", {
            "StartingInstances": [{"InstanceId": "i-0123456789abcdef0",
                                    "CurrentState": {"Name": "pending"},
                                    "PreviousState": {"Name": "stopped"}}]
        }, {"InstanceIds": ["i-0123456789abcdef0"]})

        monkeypatch.setattr(L1, "_WAITER_DELAY_SECONDS", 0)

        with stubber:
            result = L1.handler(_s3_event("savi-raw-006840014780", "input/ameshousing.csv"), None)

        assert result["results"][0]["action"] == "waited_then_started"
        stubber.assert_no_pending_responses()

    @pytest.mark.parametrize("state", ["pending", "running"])
    def test_pending_or_running_only_tags(self, monkeypatch, state):
        self._env(monkeypatch)
        ec2 = boto3.client("ec2", region_name="us-east-1")
        stubber = Stubber(ec2)
        monkeypatch.setattr(L1, "_ec2_client", lambda: ec2)

        stubber.add_response("create_tags", {})
        stubber.add_response("describe_instances", {
            "Reservations": [{"Instances": [{"State": {"Name": state}}]}]
        }, {"InstanceIds": ["i-0123456789abcdef0"]})

        with stubber:
            result = L1.handler(_s3_event("savi-raw-006840014780", "input/ameshousing.csv"), None)

        assert result["results"][0]["action"] == "tags_only"
        stubber.assert_no_pending_responses()

    @pytest.mark.parametrize("state", ["terminated", "shutting-down"])
    def test_terminal_states_raise(self, monkeypatch, state):
        self._env(monkeypatch)
        ec2 = boto3.client("ec2", region_name="us-east-1")
        stubber = Stubber(ec2)
        monkeypatch.setattr(L1, "_ec2_client", lambda: ec2)

        stubber.add_response("create_tags", {})
        stubber.add_response("describe_instances", {
            "Reservations": [{"Instances": [{"State": {"Name": state}}]}]
        }, {"InstanceIds": ["i-0123456789abcdef0"]})

        with stubber:
            with pytest.raises(RuntimeError):
                L1.handler(_s3_event("savi-raw-006840014780", "input/ameshousing.csv"), None)

    def test_irrelevant_key_produces_no_api_calls(self, monkeypatch):
        self._env(monkeypatch)
        ec2 = boto3.client("ec2", region_name="us-east-1")
        stubber = Stubber(ec2)
        monkeypatch.setattr(L1, "_ec2_client", lambda: ec2)

        with stubber:
            result = L1.handler(_s3_event("savi-raw-006840014780", "reference/foo.csv"), None)

        assert result["processed"] == 0
        stubber.assert_no_pending_responses()


# ════════════════════════════════════════════════════════════════════
# Lambda 2 — lambda_start_sagemaker
# ════════════════════════════════════════════════════════════════════
class TestSuccessKeyParsing:
    def test_extracts_run_id(self):
        assert L2.extract_run_id_from_key("runs/20261002T210501Z-ameshousing/_SUCCESS.json") == \
            "20261002T210501Z-ameshousing"

    def test_rejects_wrong_prefix(self):
        assert L2.extract_run_id_from_key("other/20261002T210501Z-ameshousing/_SUCCESS.json") is None

    def test_rejects_wrong_suffix(self):
        assert L2.extract_run_id_from_key("runs/20261002T210501Z-ameshousing/other.json") is None

    def test_rejects_nested_path(self):
        assert L2.extract_run_id_from_key("runs/foo/bar/_SUCCESS.json") is None


class TestJobNameSanitization:
    def test_basic_name(self):
        name = L2.sanitize_job_name("20261002T210501Z-ameshousing")
        assert name == "savi-dqn-20261002T210501Z-ameshousing"
        assert len(name) <= 63

    def test_long_run_id_is_truncated(self):
        run_id = "x" * 100
        name = L2.sanitize_job_name(run_id)
        assert len(name) <= 63
        assert name.startswith("savi-dqn-")

    def test_matches_required_pattern(self):
        import re
        name = L2.sanitize_job_name("20261002T210501Z-ames housing!!")
        assert re.match(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$", name)

    def test_empty_stem_falls_back(self):
        name = L2.sanitize_job_name("!!!")
        assert name == "savi-dqn-run"


_ENV = {
    "ROLE_ARN": "arn:aws:iam::006840014780:role/LabRole",
    "IMAGE_URI": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-sagemaker",
    "CPU_IMAGE_URI": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-cpu-py312-ubuntu22.04-sagemaker",
    "INSTANCE_TYPE": "ml.g4dn.xlarge",
    "FALLBACK_INSTANCE_TYPE": "ml.m5.xlarge",
    "USE_SPOT": "true",
    "CODE_S3_URI": "s3://savi-processed-006840014780/code/sourcedir.tar.gz",
    "PROCESSED_BUCKET": "savi-processed-006840014780",
    "MAX_RUNTIME": "3600",
    "EPOCHS": "150",
    "REGION": "us-east-1",
}


class TestBuildTrainingJobRequest:
    def test_hyperparameters_are_json_encoded_strings(self):
        req = L2.build_training_job_request("run-123", "ml.g4dn.xlarge", _ENV["IMAGE_URI"], True, _ENV)
        hp = req["HyperParameters"]
        assert hp["sagemaker_program"] == json.dumps("savi_gpu_sagemaker.py")
        assert hp["epochs"] == json.dumps(150)
        assert hp["run-id"] == json.dumps("run-123")
        assert hp["sagemaker_container_log_level"] == json.dumps(20)
        assert hp["sagemaker_submit_directory"] == json.dumps(_ENV["CODE_S3_URI"])
        for v in hp.values():
            assert isinstance(v, str)
            json.loads(v)  # debe ser JSON válido

    def test_spot_fields_present_when_spot(self):
        req = L2.build_training_job_request("run-123", "ml.g4dn.xlarge", _ENV["IMAGE_URI"], True, _ENV)
        assert req["EnableManagedSpotTraining"] is True
        assert req["StoppingCondition"]["MaxWaitTimeInSeconds"] == 2 * 3600
        assert req["CheckpointConfig"]["S3Uri"] == "s3://savi-processed-006840014780/runs/run-123/checkpoints/"
        assert req["CheckpointConfig"]["LocalPath"] == "/opt/ml/checkpoints"

    def test_spot_fields_absent_when_on_demand(self):
        req = L2.build_training_job_request("run-123", "ml.g4dn.xlarge", _ENV["IMAGE_URI"], False, _ENV)
        assert "EnableManagedSpotTraining" not in req
        assert "CheckpointConfig" not in req
        assert "MaxWaitTimeInSeconds" not in req["StoppingCondition"]

    def test_channel_and_output_paths(self):
        req = L2.build_training_job_request("run-123", "ml.g4dn.xlarge", _ENV["IMAGE_URI"], True, _ENV)
        channel = req["InputDataConfig"][0]
        assert channel["ChannelName"] == "processed"
        s3src = channel["DataSource"]["S3DataSource"]
        assert s3src["S3Uri"] == "s3://savi-processed-006840014780/runs/run-123/"
        assert s3src["S3DataType"] == "S3Prefix"
        assert s3src["S3DataDistributionType"] == "FullyReplicated"
        assert req["AlgorithmSpecification"]["TrainingInputMode"] == "File"
        assert req["OutputDataConfig"]["S3OutputPath"] == "s3://savi-processed-006840014780/runs/run-123/sagemaker/"

    def test_resource_config(self):
        req = L2.build_training_job_request("run-123", "ml.g4dn.xlarge", _ENV["IMAGE_URI"], True, _ENV)
        assert req["ResourceConfig"]["InstanceType"] == "ml.g4dn.xlarge"
        assert req["ResourceConfig"]["InstanceCount"] == 1
        assert req["ResourceConfig"]["VolumeSizeInGB"] == 10

    def test_tags(self):
        req = L2.build_training_job_request("run-123", "ml.g4dn.xlarge", _ENV["IMAGE_URI"], True, _ENV)
        tag_dict = {t["Key"]: t["Value"] for t in req["Tags"]}
        assert tag_dict == {"Project": "SAVI", "RunId": "run-123"}


class TestSuccessStatusParsing:
    def _stub_s3_get_object(self, body: dict):
        import io
        from botocore.response import StreamingBody
        payload = json.dumps(body).encode("utf-8")
        s3 = boto3.client("s3", region_name="us-east-1")
        stubber = Stubber(s3)
        stubber.add_response("get_object", {
            "Body": StreamingBody(io.BytesIO(payload), len(payload))
        })
        return s3, stubber

    def test_succeeded_proceeds(self, monkeypatch):
        s3, stubber = self._stub_s3_get_object({"run_id": "run-123", "status": "SUCCEEDED"})
        sm = boto3.client("sagemaker", region_name="us-east-1")
        sm_stubber = Stubber(sm)
        sm_stubber.add_response("create_training_job", {"TrainingJobArn": "arn:aws:sagemaker:..."})

        record = {"s3": {"bucket": {"name": "savi-processed-006840014780"},
                          "object": {"key": "runs/run-123/_SUCCESS.json"}}}
        with stubber, sm_stubber:
            result = L2._handle_record(sm, s3, record, _ENV)

        assert result["status"] == "created"
        assert result["run_id"] == "run-123"

    def test_non_succeeded_status_is_skipped(self):
        s3, stubber = self._stub_s3_get_object({"run_id": "run-123", "status": "FAILED"})
        sm = boto3.client("sagemaker", region_name="us-east-1")
        sm_stubber = Stubber(sm)  # sin respuestas: no debe llamarse create_training_job

        record = {"s3": {"bucket": {"name": "savi-processed-006840014780"},
                          "object": {"key": "runs/run-123/_SUCCESS.json"}}}
        with stubber, sm_stubber:
            result = L2._handle_record(sm, s3, record, _ENV)

        assert result["status"] == "skipped"
        sm_stubber.assert_no_pending_responses()


class TestFallbackChain:
    def _client_error(self, code: str, message: str = "boom"):
        return ClientError({"Error": {"Code": code, "Message": message}}, "CreateTrainingJob")

    def test_spot_succeeds_first_try(self):
        sm = boto3.client("sagemaker", region_name="us-east-1")
        stubber = Stubber(sm)
        stubber.add_response("create_training_job", {"TrainingJobArn": "arn:..."})
        with stubber:
            result = L2._create_with_fallback(sm, "run-123", _ENV)
        assert result["use_spot"] is True
        assert result["instance_type"] == "ml.g4dn.xlarge"
        assert result["attempt"] == 1
        stubber.assert_no_pending_responses()

    def test_spot_capacity_error_falls_back_to_on_demand_same_type(self):
        sm = boto3.client("sagemaker", region_name="us-east-1")
        stubber = Stubber(sm)
        stubber.add_client_error("create_training_job", service_error_code="ResourceLimitExceeded")
        stubber.add_response("create_training_job", {"TrainingJobArn": "arn:..."})
        with stubber:
            result = L2._create_with_fallback(sm, "run-123", _ENV)
        assert result["use_spot"] is False
        assert result["instance_type"] == "ml.g4dn.xlarge"
        assert result["attempt"] == 2
        stubber.assert_no_pending_responses()

    def test_validation_exception_mentioning_spot_falls_back(self):
        sm = boto3.client("sagemaker", region_name="us-east-1")
        stubber = Stubber(sm)
        stubber.add_client_error("create_training_job", service_error_code="ValidationException",
                                  service_message="Spot instances are not supported for this configuration")
        stubber.add_response("create_training_job", {"TrainingJobArn": "arn:..."})
        with stubber:
            result = L2._create_with_fallback(sm, "run-123", _ENV)
        assert result["use_spot"] is False
        stubber.assert_no_pending_responses()

    def test_falls_back_all_the_way_to_cpu_fallback_instance(self):
        sm = boto3.client("sagemaker", region_name="us-east-1")
        stubber = Stubber(sm)
        stubber.add_client_error("create_training_job", service_error_code="ResourceLimitExceeded")
        stubber.add_client_error("create_training_job", service_error_code="CapacityError")
        stubber.add_client_error("create_training_job", service_error_code="ResourceLimitExceeded")
        stubber.add_response("create_training_job", {"TrainingJobArn": "arn:..."})
        with stubber:
            result = L2._create_with_fallback(sm, "run-123", _ENV)
        assert result["instance_type"] == "ml.m5.xlarge"
        assert result["use_spot"] is False
        assert result["attempt"] == 4
        stubber.assert_no_pending_responses()

    def test_learner_lab_gpu_access_denied_falls_to_cpu_spot(self):
        """Learner Lab niega ml.g4dn.* por política IAM explícita → CPU con Spot."""
        sm = boto3.client("sagemaker", region_name="us-east-1")
        stubber = Stubber(sm)
        stubber.add_client_error("create_training_job", service_error_code="AccessDeniedException",
                                 service_message="explicit deny in an identity-based policy")
        stubber.add_client_error("create_training_job", service_error_code="AccessDeniedException",
                                 service_message="explicit deny in an identity-based policy")
        stubber.add_response("create_training_job", {"TrainingJobArn": "arn:..."})
        with stubber:
            result = L2._create_with_fallback(sm, "run-123", _ENV)
        assert result["instance_type"] == "ml.m5.xlarge"
        assert result["use_spot"] is True
        assert result["attempt"] == 3
        stubber.assert_no_pending_responses()

    def test_resource_in_use_is_idempotent_success(self):
        sm = boto3.client("sagemaker", region_name="us-east-1")
        stubber = Stubber(sm)
        stubber.add_client_error("create_training_job", service_error_code="ResourceInUse")
        with stubber:
            result = L2._create_with_fallback(sm, "run-123", _ENV)
        assert result["status"] == "already_exists"
        stubber.assert_no_pending_responses()

    def test_unrecoverable_error_propagates(self):
        sm = boto3.client("sagemaker", region_name="us-east-1")
        stubber = Stubber(sm)
        stubber.add_client_error("create_training_job", service_error_code="ValidationException",
                                 service_message="Invalid hyperparameter value")
        with stubber:
            with pytest.raises(ClientError):
                L2._create_with_fallback(sm, "run-123", _ENV)
        stubber.assert_no_pending_responses()

    def test_no_spot_env_skips_spot_attempt(self):
        env = dict(_ENV)
        env["USE_SPOT"] = "false"
        sm = boto3.client("sagemaker", region_name="us-east-1")
        stubber = Stubber(sm)
        stubber.add_response("create_training_job", {"TrainingJobArn": "arn:..."})
        with stubber:
            result = L2._create_with_fallback(sm, "run-123", env)
        assert result["use_spot"] is False
        assert result["instance_type"] == "ml.g4dn.xlarge"
        assert result["attempt"] == 1
        stubber.assert_no_pending_responses()
