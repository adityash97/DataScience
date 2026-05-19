import logging

logger = logging.getLogger(__name__)

name = 'calculator'
description = 'Perform basic arithmetic. Input: {"op": "add"|"subtract"|"multiply"|"divide", "a": float, "b": float}.'

_OPS = {
    'add': lambda a, b: a + b,
    'subtract': lambda a, b: a - b,
    'multiply': lambda a, b: a * b,
    'divide': lambda a, b: a / b,
}


def run(payload: dict) -> dict:
    payload = payload or {}
    op = payload.get('op')
    a = payload.get('a')
    b = payload.get('b')
    logger.info('[calculator_tool] %s(%s, %s)', op, a, b)

    if op not in _OPS:
        return {'tool': name, 'success': False, 'error': f"Unsupported op '{op}'. Use: {list(_OPS)}"}
    if a is None or b is None:
        return {'tool': name, 'success': False, 'error': 'Both operands a and b are required.'}

    try:
        a_f, b_f = float(a), float(b)
        if op == 'divide' and b_f == 0:
            return {'tool': name, 'success': False, 'error': 'Division by zero.'}
        result = _OPS[op](a_f, b_f)
        return {'tool': name, 'success': True, 'op': op, 'a': a_f, 'b': b_f, 'result': result}
    except (TypeError, ValueError) as exc:
        return {'tool': name, 'success': False, 'error': f'Invalid operand: {exc}'}
