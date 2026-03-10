These fixtures support the DOCX run-preserving replacement engine tests.

Files:
- original_resume.json: Baseline ResumeJSON used to create the test DOCX.
- tailored_resume_one_change.json: Same structure as the original, with one bullet text changed.
- tailored_resume_bad_structure.json: Intentionally invalid structure (bullet_id mismatch) to verify rejection.
