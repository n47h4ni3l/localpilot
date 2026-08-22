def run(self, command_spec: CommandSpec):
    if command_spec.risk == OperationRisk.DESTRUCTIVE and not self.approval_callback:
        raise RuntimeError('Destructive operation not approved')
    if command_spec.risk == OperationRisk.DESTRUCTIVE and not self.approval_callback():
        raise RuntimeError('Destructive operation not approved')
    try:
        result = subprocess.run(command_spec.argv, timeout=command_spec.timeout, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        return {'stdout': result.stdout.decode(), 'stderr': result.stderr.decode(), 'returncode': result.returncode}
    except subprocess.TimeoutExpired as e:
        return {'stdout': e.stdout.decode() if e.stdout else '', 'stderr': e.stderr.decode() if e.stderr else '', 'returncode': -1}