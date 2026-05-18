import logging

from rest_framework.decorators import api_view
from rest_framework.request import Request

from utils.response import success_response

logger = logging.getLogger(__name__)


@api_view(['GET'])
def health_check(request: Request):
    logger.debug('Health check called')
    return success_response(data={'status': 'ok'}, message='Service is healthy')
