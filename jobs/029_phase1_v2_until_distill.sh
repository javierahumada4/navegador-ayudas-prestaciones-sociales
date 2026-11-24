# est-minutes: 60
# First half of job 030 (generate + distill of the rule-sheet v2 run) to use the GPU time left
# in a session too short for the whole run. Same command and run directory as 030, stopped after
# distillation; distillation streams its traces, so a deadline cut resumes in the next session
# and job 030 then continues from SFT.
THROUGH=distill exec bash jobs/030_phase1_v2_full_run.sh
