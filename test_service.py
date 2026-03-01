# test_service.py
from backend.core.qa_service import QAService

qa = QAService()

response = qa.ask("How many days of leave do full-time employees get?")

print(response)