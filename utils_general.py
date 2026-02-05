import logging
import pickle
from typing import Optional

cache_llm = pickle.load(open("./cache_llm.pkl", "rb"))

####tmd他这是个空的储存函数#####
def save_to_cache(key: str, value: str):
    return None

###不改，只用来储存和读取缓存###
def get_from_cache(key: str) -> Optional[str]:
    try:
        return cache_llm[key.encode()].decode()
    except Exception as e:
        logging.warning(f"Error getting from cache: {e}")
    return None
