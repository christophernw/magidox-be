import os

from bsbi import BSBIIndex
from compression import VBEPostings
from letor import LambdaMart

def generate_bsbi_letor():
  output_dir = os.path.join(os.getcwd(), "index")
  data_dir = os.path.join(os.getcwd(), "collection")
  BSBI_instance = BSBIIndex(data_dir=data_dir,
                            postings_encoding=VBEPostings,
                            output_dir=output_dir)
  
  letor_instance = LambdaMart()
  # BSBI_instance.do_indexing()  
  BSBI_instance.load()

  return BSBI_instance, letor_instance

if __name__ == '__main__':
    generate_bsbi_letor()