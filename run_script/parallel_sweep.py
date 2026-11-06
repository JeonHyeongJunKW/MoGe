"""Launch independent training jobs with isolated CUDA visibility and logs."""
import json
import os
import signal
import subprocess
import time


def run_parallel(jobs, root, cwd, success_status, summarize):
    active = []
    runs = []
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        (root / 'parallel_jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
        for job in jobs:
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=job['device'], PYTHONUNBUFFERED='1')
            log = (root / (job['name'] + '.launch.log')).open('w')
            try:
                proc = subprocess.Popen(job['command'], cwd=cwd, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            finally:
                log.close()
            active.append((job, proc))
            print(f"Started {job['name']}: PID={proc.pid}, device={job['device']}, log={root / (job['name'] + '.launch.log')}", flush=True)
        pending = list(active)
        while pending:
            for job, proc in pending[:]:
                code = proc.poll()
                if code is None:
                    continue
                runs.append((job['name'], job['rank'], success_status if code == 0 else f'failed({code})'))
                pending.remove((job, proc))
                summarize(root, runs)
            if pending:
                time.sleep(1)
        return runs
    finally:
        # Interrupting the launcher must also stop Accelerate/trainer descendants.
        for _, proc in active:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 10
        for _, proc in active:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=max(.01, deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
        signal.signal(signal.SIGTERM, previous_handler)
