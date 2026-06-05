BEGIN;

WITH ranked AS (
    SELECT
        ctid,
        ROW_NUMBER() OVER (
            PARTITION BY scan_id
            ORDER BY date_created DESC NULLS LAST, id DESC
        ) AS rn
    FROM public.scan_results
    WHERE scan_id IS NOT NULL
)
DELETE FROM public.scan_results AS sr
USING ranked
WHERE sr.ctid = ranked.ctid
  AND ranked.rn > 1;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'scan_results_scan_id_unique'
          AND conrelid = 'public.scan_results'::regclass
    ) THEN
        ALTER TABLE public.scan_results
            ADD CONSTRAINT scan_results_scan_id_unique UNIQUE (scan_id);
    END IF;
END $$;

COMMIT;
