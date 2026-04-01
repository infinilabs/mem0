import logging
import time
from typing import Any, Dict, List, Optional

try:
    from easysearch import Easysearch
    from easysearch.exceptions import NotFoundError
    from easysearch.helpers import bulk
except ImportError:
    raise ImportError("Easysearch requires extra dependencies. Install with `pip install easysearch`") from None

from pydantic import BaseModel

from mem0.configs.vector_stores.easysearch import EasysearchConfig
from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: str
    score: float
    payload: Dict


class EasysearchDB(VectorStoreBase):
    def __init__(self, **kwargs):
        config = EasysearchConfig(**kwargs)

        # Initialize Easysearch client
        hosts = [{"host": config.host, "port": config.port}]
        client_kwargs: Dict[str, Any] = {
            "hosts": hosts,
            "use_ssl": config.use_ssl,
            "verify_certs": config.verify_certs,
        }
        if config.user and config.password:
            client_kwargs["http_auth"] = (config.user, config.password)

        self.client = Easysearch(**client_kwargs)

        self.collection_name = config.collection_name
        self.embedding_model_dims = config.embedding_model_dims
        self.create_col(self.collection_name, self.embedding_model_dims)

    def create_col(self, name: str, vector_size: int, distance: str = "cosine") -> None:
        """Create a new collection (index in Easysearch)."""
        index_settings = {
            "mappings": {
                "properties": {
                    "vector_field": {
                        "type": "knn_dense_float_vector",
                        "knn": {
                            "dims": vector_size,
                            "model": "lsh",
                            "similarity": "cosine",
                            "L": 99,
                            "k": 1,
                        },
                    },
                    "payload": {"type": "object"},
                    "id": {"type": "keyword"},
                }
            },
        }

        if not self.client.indices.exists(index=name):
            logger.warning(f"Creating index {name}, it might take 1-2 minutes...")
            self.client.indices.create(index=name, body=index_settings)

            # Wait for index to be ready
            max_retries = 180
            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.client.search(index=name, body={"query": {"match_all": {}}})
                    time.sleep(1)
                    logger.info(f"Index {name} is ready")
                    return
                except Exception:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise TimeoutError(f"Index {name} creation timed out after {max_retries} seconds")
                    time.sleep(0.5)

    def insert(
        self, vectors: List[List[float]], payloads: Optional[List[Dict]] = None, ids: Optional[List[str]] = None
    ) -> List[OutputData]:
        """Insert vectors into the index."""
        if not ids:
            ids = [str(i) for i in range(len(vectors))]

        if payloads is None:
            payloads = [{} for _ in range(len(vectors))]

        for idx, vec in enumerate(vectors):
            if vec is None:
                raise ValueError(
                    f"Vector at index {idx} is null. "
                    f"This usually means the embedding model failed to generate an embedding. "
                    f"Check that your embedding model is configured correctly and returning valid vectors."
                )
            if len(vec) == 0:
                raise ValueError(
                    f"Vector at index {idx} is empty. "
                    f"Expected a vector of dimension {self.embedding_model_dims}, got an empty vector."
                )
            if len(vec) != self.embedding_model_dims:
                raise ValueError(
                    f"Vector at index {idx} has dimension {len(vec)}, "
                    f"but the index '{self.collection_name}' expects dimension {self.embedding_model_dims}. "
                    f"Ensure your embedding model's output dimensions match the vector store configuration."
                )

        actions = []
        for i, (vec, id_) in enumerate(zip(vectors, ids)):
            action = {
                "_index": self.collection_name,
                "_id": id_,
                "_source": {
                    "vector_field": vec,
                    "payload": payloads[i],
                    "id": id_,
                },
            }
            actions.append(action)

        bulk(self.client, actions)
        self.client.indices.refresh(index=self.collection_name)

        results = []
        for i, id_ in enumerate(ids):
            results.append(OutputData(id=id_, score=1.0, payload=payloads[i]))
        return results

    def search(
        self, query: str, vectors: List[float], limit: int = 5, filters: Optional[Dict] = None
    ) -> List[OutputData]:
        """Search for similar vectors using Easysearch knn_nearest_neighbors query."""
        knn_query = {
            "knn_nearest_neighbors": {
                "field": "vector_field",
                "vec": {"values": vectors},
                "model": "lsh",
                "similarity": "cosine",
                "candidates": limit * 2,
            }
        }

        query_body: Dict[str, Any] = {"size": limit, "query": None}

        filter_clauses = []
        if filters:
            for key in ["user_id", "run_id", "agent_id"]:
                value = filters.get(key)
                if value:
                    filter_clauses.append({"term": {f"payload.{key}.keyword": value}})

        if filter_clauses:
            query_body["query"] = {"bool": {"must": knn_query, "filter": filter_clauses}}
        else:
            query_body["query"] = knn_query

        try:
            response = self.client.search(index=self.collection_name, body=query_body)

            hits = response["hits"]["hits"]
            results = [
                OutputData(id=hit["_source"].get("id", hit["_id"]), score=hit["_score"], payload=hit["_source"].get("payload", {}))
                for hit in hits[:limit]
            ]
            return results
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            return []

    def delete(self, vector_id: str) -> None:
        """Delete a vector by ID."""
        self.client.delete(index=self.collection_name, id=vector_id)

    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict] = None) -> None:
        """Update a vector and its payload."""
        if vector is not None:
            if len(vector) == 0:
                raise ValueError("Cannot update with an empty vector.")
            if len(vector) != self.embedding_model_dims:
                raise ValueError(
                    f"Update vector has dimension {len(vector)}, "
                    f"but the index '{self.collection_name}' expects dimension {self.embedding_model_dims}. "
                    f"Ensure your embedding model's output dimensions match the vector store configuration."
                )

        doc = {}
        if vector is not None:
            doc["vector_field"] = vector
        if payload is not None:
            doc["payload"] = payload

        if doc:
            try:
                self.client.update(index=self.collection_name, id=vector_id, body={"doc": doc})
            except Exception as e:
                logger.error(f"Error updating vector {vector_id}: {e}", exc_info=True)
                raise

    def get(self, vector_id: str) -> Optional[OutputData]:
        """Retrieve a vector by ID."""
        try:
            response = self.client.get(index=self.collection_name, id=vector_id)
            return OutputData(
                id=response["_id"],
                score=1.0,
                payload=response["_source"].get("payload", {}),
            )
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error retrieving vector {vector_id}: {str(e)}", exc_info=True)
            return None

    def list_cols(self) -> List[str]:
        """List all collections (indices)."""
        return list(self.client.indices.get_alias().keys())

    def delete_col(self) -> None:
        """Delete a collection (index)."""
        self.client.indices.delete(index=self.collection_name)

    def col_info(self, name=None) -> Any:
        """Get information about a collection (index)."""
        return self.client.indices.get(index=name)

    def list(self, filters: Optional[Dict] = None, limit: Optional[int] = None) -> List[List[OutputData]]:
        """List all memories with optional filters."""
        try:
            query: Dict = {"query": {"match_all": {}}}

            filter_clauses = []
            if filters:
                for key in ["user_id", "run_id", "agent_id"]:
                    value = filters.get(key)
                    if value:
                        filter_clauses.append({"term": {f"payload.{key}.keyword": value}})

            if filter_clauses:
                query["query"] = {"bool": {"filter": filter_clauses}}

            if limit:
                query["size"] = limit

            response = self.client.search(index=self.collection_name, body=query)
            hits = response["hits"]["hits"]

            results = [
                OutputData(id=hit["_source"].get("id", hit["_id"]), score=1.0, payload=hit["_source"].get("payload", {}))
                for hit in hits
            ]
            return [results]
        except Exception as e:
            logger.error(f"Error listing vectors: {e}", exc_info=True)
            return []

    def reset(self):
        """Reset the index by deleting and recreating it."""
        logger.warning(f"Resetting index {self.collection_name}...")
        self.delete_col()
        self.create_col(self.collection_name, self.embedding_model_dims)
