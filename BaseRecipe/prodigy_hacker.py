import os
from urllib.parse import urlparse, parse_qs

from prodigy.app import app as prodigy_app, _shared_get_questions
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import request_response


def start_hacking(prodigy_data_provider_by_doi):
    def hacky_get_questions(request: Request):
        referer = request.headers.get("referer")
        query = parse_qs(urlparse(referer).query)
        doi = query.get('doi', None)
        if isinstance(doi, list) and len(doi) == 1:
            doi = doi[0]
            prodigy_data = list(prodigy_data_provider_by_doi(doi))
            result = {
                'total': 1,
                'progress': None,
                'session_id': None,
                'tasks': prodigy_data
            }
            return JSONResponse(result)
        elif not isinstance(doi, str):
            doi = 'No doi!'
            return JSONResponse(_shared_get_questions(None))

    with open(os.path.join(os.path.dirname(__file__), 'bundle.js'), encoding='utf8') as f:
        my_js = f.read()

    with open(os.path.join(os.path.dirname(__file__), 'index.html'), encoding='utf8') as f:
        my_html = f.read()

    def static_bundle(_):
        return Response(my_js, media_type="application/javascript")

    def static_index(_):
        return Response(my_html, media_type="text/html")

    for i, route in enumerate(prodigy_app.router.routes):
        if route.path == '/get_session_questions':
            route.endpoint = hacky_get_questions
            route.app = request_response(hacky_get_questions)
        elif route.path == '/bundle.js':
            route.endpoint = static_bundle
            route.app = request_response(static_bundle)
        elif route.path in {'/', '/index.html'}:
            route.endpoint = static_index
            route.app = request_response(static_index)