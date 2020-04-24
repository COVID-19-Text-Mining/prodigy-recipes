from pprint import pprint
from urllib.parse import urlparse, parse_qs
from prodigy.app import app as prodigy_app, _shared_get_questions
from starlette.requests import Request
from starlette.responses import JSONResponse
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


    for i, route in enumerate(prodigy_app.router.routes):
        if route.path == '/get_session_questions':
            route.endpoint = hacky_get_questions
            route.app = request_response(hacky_get_questions)
            break
