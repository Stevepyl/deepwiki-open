from fastapi.testclient import TestClient

from api import api as api_module


client = TestClient(api_module.app)


def _wiki_cache_payload():
    return {
        "repo": {"owner": "infiniflow", "repo": "ragflow", "type": "github"},
        "language": "en",
        "wiki_structure": {
            "id": "wiki",
            "title": "Project Wiki",
            "description": "Generated repository wiki",
            "pages": [
                {
                    "id": "troubleshooting",
                    "title": "Common Issues & Troubleshooting",
                    "content": "",
                    "filePaths": ["README.md"],
                    "importance": "high",
                    "relatedPages": [],
                }
            ],
            "sections": [{"id": "contents", "title": "Contents", "pages": ["troubleshooting"]}],
        },
        "generated_pages": {"troubleshooting": "# Common Issues\ncontent"},
    }


def test_wiki_cache_accepts_frontend_markdown_page_payload(monkeypatch):
    saved = {}

    async def fake_save_wiki_cache(request_data):
        saved["generated_page"] = request_data.generated_pages["troubleshooting"]
        saved["provider"] = request_data.provider
        saved["model"] = request_data.model
        return True

    monkeypatch.setattr(api_module, "save_wiki_cache", fake_save_wiki_cache)

    response = client.post("/api/wiki_cache", json=_wiki_cache_payload())

    assert response.status_code == 200, response.text
    assert response.json() == {"message": "Wiki cache saved successfully"}
    assert saved == {
        "generated_page": "# Common Issues\ncontent",
        "provider": None,
        "model": None,
    }
