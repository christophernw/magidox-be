import os
from typing import Any, Optional
from pydantic import BaseModel
from typing import List


class SearchQuery(BaseModel):
  content: str
  k: Optional[int] = 10
  rerank: Optional[bool]


class DocsQuery(BaseModel):
  part: str
  cid: str


class Result(BaseModel):
  def __init__(self, score: int, path: str, **data: Any):
    super().__init__(**data)
    print(path)
    self.score = score
    self.path = path.split('collection')[-1][1:]
    print(self.path)
    self.id = path.split('\\')[-1]
    self.excerpt = ""
    with(open(path, "r")) as buffer:
      for i in buffer:
        words = i.split()
        for word in words:
          self.excerpt += word + " "
          if (len(self.excerpt) > 250):
            self.excerpt += "..."
            break

  path: Optional[str] = ""
  score: Optional[int] = 0
  id: Optional[str] = ""
  excerpt: Optional[str] = ""


class SearchResponse(BaseModel):
  def __init__(self, code: int, results: List[Result], **data: Any):
    super().__init__(**data)
    self.code = code
    self.results = results

  results: Optional[List[Result]] = []
  code: Optional[int] = 500


class SpellCheckResponse(BaseModel):
  def __init__(self, code: int, spellcheck: str, changed: bool, **data: Any):
    super().__init__(**data)
    self.code = code
    self.spellcheck = spellcheck
    self.changed = changed

  spellcheck: Optional[str] = ""
  changed: Optional[bool] = False
  code: Optional[int] = 500


def get_content(part, cid):
  cur_path = os.path.join(os.getcwd(), "collection", part, cid)
  print(cur_path, "asdfasdfasdfas")
  try:
    with(open(cur_path, "r")) as buffer:
      return buffer.read()
  except:
    return ""
  
def get_content_top(part, cid):
  cur_path = os.path.join(os.getcwd(), "collection", part, cid)
  print(cur_path, "asdfasdfasdfas")
  try:
    # with(open(cur_path, "r")) as buffer:
    #   return buffer.read()[:300]
    top_results = ""
    with(open(cur_path, "r")) as buffer:
      for i in buffer:
        words = i.split()
        for word in words:
          top_results += word + " "
          if (len(top_results) > 500):
            top_results += "..."
            break
    return top_results
  except:
    return ""

def engine_to_result_list(engine_list):
  result_list = [Result(engine_result[0], engine_result[1]) for engine_result in
                 engine_list]
  return result_list


class ErrorResponse(BaseModel):
  def __init__(self, error: str, error_description: str, **data: Any):
    super().__init__(**data)
    self.error = error
    self.error_description = error_description

  error: str = "invalid_token"
  error_description: str = "An error has occured"