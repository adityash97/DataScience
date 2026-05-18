import logging

from rest_framework.response import Response
from rest_framework.views import exception_handler

logger = logging.getLogger(__name__)


def custom_exception_handler(exc, context):
    response = exception_handler(exc, context)

    if response is not None:
        logger.warning('API error: %s', exc)
        return Response(
            {
                'success': False,
                'message': str(exc),
                'errors': response.data,
            },
            status=response.status_code,
        )

    logger.exception('Unhandled exception in %s', context.get('view'))
    return None
