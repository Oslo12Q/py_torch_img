import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "py_torch_img.settings")

application = get_wsgi_application()
