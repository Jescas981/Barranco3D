"""Per-operation execution settings, independent of model/cache configuration."""
OPERATIONS = ('global', 'local', 'matching', 'dense')
FIELDS = {'batch_size', 'workers', 'loader_workers', 'prefetch', 'max_in_flight'}


def validate_execution(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) - set(OPERATIONS):
        raise ValueError(f'execution must map these operations: {OPERATIONS}')
    for operation, settings in value.items():
        if not isinstance(settings, dict) or set(settings) - FIELDS:
            raise ValueError(f'Invalid execution.{operation} settings')
        for key, number in settings.items():
            minimum = 0 if key == 'loader_workers' else 1
            if type(number) is not int or number < minimum:
                raise ValueError(f'execution.{operation}.{key} must be an integer >= {minimum}')
    return value


def execution_options(args, operation):
    """New settings override legacy flags without changing their defaults."""
    result = dict(batch_size=getattr(args, 'dense_batch_size' if operation == 'dense' else 'local_batch_size', 1)
                  if operation in ('local', 'dense') else 1,
                  workers=1, loader_workers=getattr(args, 'loader_workers', 1),
                  prefetch=getattr(args, 'prefetch', 4))
    result.update(validate_execution(getattr(args, 'execution', {})).get(operation, {}))
    result.setdefault('max_in_flight', result['workers'])
    return result
