BEGIN;

-- Phase 2 baseline hardening migration.
-- Forward-only and idempotent.
--
-- Duplicate handling policy for employee_baselines before uniqueness:
-- 1. Prefer the row with the highest eligible_scan_count.
-- 2. If tied, prefer the most recently updated row.
-- 3. If still tied, prefer the newest id deterministically.
-- This migration does not mutate or detach duplicates automatically.
-- If duplicates exist, it raises an error and requires explicit remediation before the unique index is added.

ALTER TABLE public.employee_baselines
    ADD COLUMN IF NOT EXISTS baseline_status varchar,
    ADD COLUMN IF NOT EXISTS eligible_scan_count integer,
    ADD COLUMN IF NOT EXISTS baseline_confidence numeric,
    ADD COLUMN IF NOT EXISTS baseline_version varchar,
    ADD COLUMN IF NOT EXISTS baseline_metadata json,
    ADD COLUMN IF NOT EXISTS last_eligible_scan_at timestamp with time zone;

ALTER TABLE public.scan_results
    ADD COLUMN IF NOT EXISTS result_status varchar,
    ADD COLUMN IF NOT EXISTS capture_quality_score numeric,
    ADD COLUMN IF NOT EXISTS measurement_reliability_score numeric,
    ADD COLUMN IF NOT EXISTS observed_fatigue_score numeric,
    ADD COLUMN IF NOT EXISTS personal_deviation_score numeric,
    ADD COLUMN IF NOT EXISTS task_completion_status varchar,
    ADD COLUMN IF NOT EXISTS baseline_status_at_inference varchar,
    ADD COLUMN IF NOT EXISTS baseline_confidence numeric,
    ADD COLUMN IF NOT EXISTS baseline_eligible boolean,
    ADD COLUMN IF NOT EXISTS hard_gates_triggered json,
    ADD COLUMN IF NOT EXISTS explainable_reasons json;

UPDATE public.employee_baselines
SET eligible_scan_count = COALESCE(eligible_scan_count, scan_count, 0)
WHERE eligible_scan_count IS NULL;

UPDATE public.employee_baselines
SET baseline_status = COALESCE(
        baseline_status,
        CASE
            WHEN COALESCE(eligible_scan_count, 0) >= 5 THEN 'active'
            WHEN COALESCE(eligible_scan_count, 0) >= 3 THEN 'provisional'
            ELSE 'collecting'
        END
    )
WHERE baseline_status IS NULL;

UPDATE public.employee_baselines
SET baseline_confidence = COALESCE(
        baseline_confidence,
        CASE
            WHEN COALESCE(eligible_scan_count, 0) <= 0 THEN 0
            WHEN COALESCE(eligible_scan_count, 0) >= 5 THEN 1
            ELSE ROUND((COALESCE(eligible_scan_count, 0)::numeric / 5.0), 4)
        END
    )
WHERE baseline_confidence IS NULL;

UPDATE public.employee_baselines
SET baseline_version = COALESCE(baseline_version, 'robust_v2')
WHERE baseline_version IS NULL;

UPDATE public.employee_baselines
SET baseline_metadata = COALESCE(
        baseline_metadata,
        json_build_object(
            'schema_version', 2,
            'source', 'qualified_calibration_scans'
        )
    )
WHERE baseline_metadata IS NULL;

UPDATE public.scan_results
SET baseline_eligible = COALESCE(baseline_eligible, FALSE)
WHERE baseline_eligible IS NULL;

ALTER TABLE public.employee_baselines
    DROP CONSTRAINT IF EXISTS employee_baselines_baseline_status_check;

ALTER TABLE public.employee_baselines
    ADD CONSTRAINT employee_baselines_baseline_status_check
    CHECK (
        baseline_status IS NULL
        OR baseline_status IN ('collecting', 'provisional', 'active', 'needs_review', 'disabled')
    );

ALTER TABLE public.scan_results
    DROP CONSTRAINT IF EXISTS scan_results_result_status_check;

ALTER TABLE public.scan_results
    ADD CONSTRAINT scan_results_result_status_check
    CHECK (
        result_status IS NULL
        OR result_status IN ('scored', 'retake_required', 'incomplete', 'low_confidence', 'failed')
    );

ALTER TABLE public.scan_results
    DROP CONSTRAINT IF EXISTS scan_results_task_completion_status_check;

ALTER TABLE public.scan_results
    ADD CONSTRAINT scan_results_task_completion_status_check
    CHECK (
        task_completion_status IS NULL
        OR task_completion_status IN ('completed', 'incomplete_required_speech', 'incomplete_required_task', 'not_required')
    );

ALTER TABLE public.scan_results
    DROP CONSTRAINT IF EXISTS scan_results_risk_level_phase2_check;

ALTER TABLE public.scan_results
    ADD CONSTRAINT scan_results_risk_level_phase2_check
    CHECK (
        risk_level IS NULL
        OR risk_level IN ('stable', 'low_focus', 'elevated_fatigue', 'high_risk', 'unknown')
    );

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM public.employee_baselines
        WHERE business_profile IS NOT NULL
          AND member IS NOT NULL
        GROUP BY business_profile, member
        HAVING COUNT(*) > 1
    ) THEN
        RAISE EXCEPTION
            'Duplicate employee_baselines rows exist for (business_profile, member). Resolve duplicates manually using eligible_scan_count desc, date_updated desc, id desc retention order before applying the unique index.';
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS employee_baselines_business_profile_member_unique
    ON public.employee_baselines (business_profile, member)
    WHERE business_profile IS NOT NULL AND member IS NOT NULL;

COMMIT;
