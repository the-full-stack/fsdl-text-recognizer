import base64
import pathlib
from unittest import TestCase

from api.app import app


REPO_DIRNAME = pathlib.Path(__file__).parents[2].resolve()
SUPPORT_DIRNAME = REPO_DIRNAME / 'text_recognizer' / 'tests'/ 'support' / 'emnist'


class TestIntegrations(TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_index(self):
        response = self.app.get('/')
        assert response.get_data().decode() == 'Hello, world!'

    def test_predict(self):
        with open(SUPPORT_DIRNAME / '0.png', 'rb') as f:
            b64_image = base64.b64encode(f.read())
        response = self.app.post('/v1/predict', json={
            'image': f'data:image/jpeg;base64,{b64_image.decode()}'
        })
        json_data = response.get_json()
        self.assertLess(abs(json_data['conf'] - 0.53), 0.01)
        self.assertEquals(json_data['pred'], '0')
