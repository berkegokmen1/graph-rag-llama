import json
from datetime import date, datetime
from neo4j.time import Date


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        elif isinstance(obj, Date):
            return obj.iso_format()
        return super().default(obj)
