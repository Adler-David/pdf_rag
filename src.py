import logging
import time
from pathlib import Path
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from torch.utils.hipify.hipify_python import meta_data
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from io import BytesIO
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from pymilvus import MilvusClient
import webbrowser
import requests
import json
from typing import List, Optional

MILVUS_IP = "http://25.12.119.117"
MILVUS_PORT = "19530"
MILVUS_LOCATION = MILVUS_IP+":"+MILVUS_PORT

def create_pdf_from_bytes(bytes, file_route):
    with open(file_route, "wb") as f:
        f.write(bytes)


def open_file_in_web(file_route):
    webbrowser.open(file_route)
    
    
def pdf_bytes_to_doc_stream(pdf_bytes, doc_name):
    buf = BytesIO(pdf_bytes)
    source = DocumentStream(name=doc_name, stream=buf)
    return source


def extract_elements_from_pdf(pdf_document, image_resolution_scale=2.0, save_pages=False, save_figures=False, save_extraction_to_md=False, save_extraction_to_html=False):
    
    # input_doc_path = "https://arxiv.org/pdf/2408.09869"
    
    _log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    pipeline_options = PdfPipelineOptions()
    
    pipeline_options.images_scale = image_resolution_scale
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    conv_res = doc_converter.convert(pdf_document)
    if save_pages:
        save_pages(conv_res)
    if save_figures:
        save_figures(conv_res)
    if save_extraction_to_md:
        save_extraction_to_md(conv_res)
    if save_extraction_to_html:
        save_extraction_to_html(conv_res)
        
    #data_folder = Path("__file__").parent / "data" #?
    return conv_res


def save_pages(conv_res):
    output_dir = Path("proceced_files")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem
    
    # Save page images
    for page_no, page in conv_res.document.pages.items():
        page_no = page.page_no
        page_image_filename = output_dir / f"{doc_filename}-{page_no}.png"
        with page_image_filename.open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")


def save_figures(conv_res):
    output_dir = Path("proceced_files")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem
    
    # Save images of figures and tables
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                    output_dir / f"{doc_filename}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")
    
        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = (
                    output_dir / f"{doc_filename}-picture-{picture_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")


def save_extraction_to_md(conv_res):
    
    output_dir = Path("proceced_files")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem
    
    # Save markdown with embedded pictures
    md_filename = output_dir / f"{doc_filename}-with-images.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)
    
    # Save markdown with externally referenced pictures
    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)
    

def save_extraction_to_html(conv_res):
    
    output_dir = Path("proceced_files")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem
    
    # Save HTML with externally referenced pictures
    html_filename = output_dir / f"{doc_filename}-with-image-refs.html"
    conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)
    

def chunk(conv_res, embedding_model, max_tokens=500):

    # EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    # MAX_TOKENS = 250

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(embedding_model),
        max_tokens=max_tokens,
    )

    doc = conv_res.document
    chunks = []
    chunker = HybridChunker(tokenizer=tokenizer,merge_peers=True)
    chunk_iter = chunker.chunk(dl_doc=doc)
    return chunk_iter

def contextualize(chunk, embedding_model, max_tokens=500):
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(embedding_model),
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(tokenizer=tokenizer,merge_peers=True)
    enriched_text = chunker.contextualize(chunk=chunk)
    return enriched_text

def instant_sentece_transformer_embedder(embedding_model):
    # embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(embedding_model)
    return model



def embed_with_ollama(text, ollama_url="http://25.12.119.117:11435", model="bge-m3", timeout=30):
    try:
        # Prepare the request payload
        payload = {
            "model": model,
            "input": text
        }
        
        print(f"Sending request to {ollama_url}/api/embeddings...")
        
        # Make the API request with timeout
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json=payload,
            # headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        
        print(f"Response status: {response.status_code}")
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response and extract the embedding
        result = response.json()
        print(result)
        embedding = result.["embedding"][0]
        
        if len(embedding) == 0:
            print("Error: No embedding found in response")
            print(f"Response content: {result}")
            return None
            
        print("Received embedding successfully")
        print(embedding)
        print(type(embedding))
        return embedding
        
    except requests.exceptions.Timeout:
        print(f"Request timed out after {timeout} seconds")
        return None
    except requests.exceptions.ConnectionError:
        print(f"Connection error - could not connect to {ollama_url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Response text: {response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def embedd(chunk, embedder):
    vector = embedder.encode(chunk)
    print(f'embedding size: {vector.shape}')
    return vector


def form_metadata_for_chunk(index, chunk):
    metadata = {
    'chunk_info': {
        'index': index,
        'text_length': len(chunk.text),
        'text_content': chunk.text,
        'items': [{"id":i.self_ref, "parent":i.parent.cref, "children":[c.cref for c in i.children], "label":i.label, "page_number":i.prov[0].page_no} for i in chunk.meta.doc_items],
        "headings": chunk.meta.headings,
        "captions": chunk.meta.captions,
        "origin_doc": chunk.meta.origin.filename 
        }
    }
    return metadata


def create_point_struct_for_qdrant(vector, metadata):
    # if hasattr(chunk, 'meta') and chunk.meta:
    #     meta_dict = {}
        
    #     if hasattr(chunk.meta, 'schema_name'):
    #         meta_dict['schema_name'] = chunk.meta.schema_name
    #     if hasattr(chunk.meta, 'version'):
    #         meta_dict['version'] = chunk.meta.version
    #     if hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
    #         doc_items = []
    #         for item in chunk.meta.doc_items:
    #             item_data = {}
                
    #             item_data['self_ref'] = getattr(item, 'self_ref', None)
    #             item_data['label'] = getattr(item, 'label', None)
                
    #             if hasattr(item, 'parent'):
    #                 item_data['parent'] = getattr(item.parent, '$ref', None) if item.parent else None
    #             item_data['children'] = getattr(item, 'children', [])
                
    #             if hasattr(item, 'prov') and item.prov:
    #                 prov_list = []
    #                 for prov in item.prov:
    #                     prov_data = {}
    #                     prov_data['page_no'] = getattr(prov, 'page_no', None)
    #                     prov_data['charspan'] = getattr(prov, 'charspan', None)
                        
    #                     if hasattr(prov, 'bbox'):
    #                         bbox = prov.bbox
    #                         prov_data['bbox'] = {
    #                             'left': getattr(bbox, 'l', None),
    #                             'top': getattr(bbox, 't', None), 
    #                             'right': getattr(bbox, 'r', None),
    #                             'bottom': getattr(bbox, 'b', None),
    #                             'coord_origin': getattr(bbox, 'coord_origin', None)
    #                         }
                        
    #                     prov_list.append(prov_data)
                    
    #                 item_data['prov'] = prov_list
                
    #             doc_items.append(item_data)
            
    #         meta_dict['doc_items'] = doc_items
        
    #     if hasattr(chunk.meta, 'headings'):
    #         meta_dict['headings'] = chunk.meta.headings
        
    #     if hasattr(chunk.meta, 'origin'):
    #         origin = chunk.meta.origin
    #         meta_dict['origin'] = {
    #             'mimetype': getattr(origin, 'mimetype', None),
    #             'filename': getattr(origin, 'filename', None), 
    #             'binary_hash': getattr(origin, 'binary_hash', None)
    #         }
        
    #     for attr in ['file_path', 'file_name', 'file_type', 'file_size', 
    #                  'creation_date', 'last_modified_date']:
    #         if hasattr(chunk.meta, attr):
    #             meta_dict[attr] = getattr(chunk.meta, attr)
        
        # metadata['docling_metadata'] = meta_dict
    
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=metadata)
    print(metadata)
    return point


# def contextualize_embedd_and_metadata(chunk_iter, embedder, embedding_model):
#     points = []
#     for index, chunk in enumerate(chunk_iter):
#         print(f"=== {index} ===")
#         print(f"chunk.text:\n{f'{chunk.text}'!r}")
#         enriched_text = contextualize(chunk, embedding_model)
#         print(f"chunker.contextualize(chunk):\n{f'{enriched_text}'!r}")
#         print()
#         vector = embedd(enriched_text, embedder)
#         metadata = form_metadata_for_chunk(index, chunk)
#         point = create_point_struct_for_qdrant(vector, metadata)
#         print(f'point metadata:{point.payload}')
#         points.append(point)
#     return points


def instant_qdarant_client():
    client = QdrantClient("localhost", port=6333)
    return client

def instant_milvus_client(milvus_uri):
    milvus_client = MilvusClient(uri=milvus_uri)
    return milvus_client




def create_qdrant_collection(collection_name, client, embedder):
    
    existing_collections = client.get_collections()
    collection_names = [collection.name for collection in existing_collections.collections]
    
    if collection_name not in collection_names:
        vector_size = embedder.get_sentence_embedding_dimension()
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Collection '{collection_name}' created successfully")
    else:
        print(f"Collection '{collection_name}' already exists")

def create_milvus_collection(milvus_client, collection_name, embedding_dim):
    if not milvus_client.has_collection(collection_name):
        milvus_client.create_collection(
            collection_name=collection_name,
            dimension=embedding_dim,
            metric_type="IP",  # Inner product distance
            consistency_level="Bounded",  # Supported values are (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`). See https://milvus.io/docs/consistency.md#Consistency-Level for more details.
        )


def insert_points_to_qdrant(client, collection_name, points):
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"inserted {len(points)} points successfully")

        
def insert_data_to_milvus(milvus_client, collection_name, data):
    milvus_client.insert(collection_name=collection_name, data=data)
    print("inserted to milvus successfully")
    
    
def pdf_to_milvus(pdf, embedding_model, collection_name):
    conv_res = extract_elements_from_pdf(pdf)
    # embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    chunks = chunk(conv_res, embedding_model)
    embedder = None
    if "sentence-transformers" in embedding_model:
        embedder = instant_sentece_transformer_embedder(embedding_model=embedding_model)
    data = []
    for index, ch in enumerate(chunks):
        print(f"=== {index} ===")
        print(f"chunk.text:\n{f'{ch.text}'!r}")
        enriched_text = contextualize(ch, embedding_model)
        print(f"chunker.contextualize(chunk):\n{f'{enriched_text}'!r}")
        print()
        if embedder:
            vector = embedd(enriched_text, embedder)
        else:
            vector = embed_with_ollama(enriched_text)
        metadata = form_metadata_for_chunk(index, ch)
        print(f'chunk metadata:{metadata}')
        data.append({"id": index, "vector": vector, "text": enriched_text, "metadata": metadata})
    client = instant_milvus_client(milvus_uri="http://localhost:19530")
    if embedder:
        embedding_dim = embedder.get_sentence_embedding_dimension()
    else:
        embedding_dim = None #to be done
    # embedding_dim = embedder.get_embedding_dimension()
    create_milvus_collection(client, collection_name, embedding_dim)
    insert_data_to_milvus(client, collection_name, data)
    
    # "http://localhost:19530"
def pdf_to_milvus_with_ollama(pdf, ollama_url, embedding_model, embedding_dim, milvus_url, collection_name):
    conv_res = extract_elements_from_pdf(pdf)
    chunks = chunk(conv_res, embedding_model)
    data = []
    for index, ch in enumerate(chunks):
        print(f"=== {index} ===")
        print(f"chunk.text:\n{f'{ch.text}'!r}")
        enriched_text = contextualize(ch, embedding_model)
        print(f"chunker.contextualize(chunk):\n{f'{enriched_text}'!r}")
        print()
        vector = embed_with_ollama(enriched_text, ollama_url, embedding_model)
        metadata = form_metadata_for_chunk(index, ch)
        print(f'chunk metadata:{metadata}')
        data.append({"id": index, "vector": vector, "text": enriched_text, "metadata": metadata})
    client = instant_milvus_client(milvus_url=milvus_url)
    create_milvus_collection(client, collection_name, embedding_dim)
    insert_data_to_milvus(client, collection_name, data)
    
    
    
def retrieve_relevent_chunks_with_sentecne_transformer(question, milvus_client=MILVUS_CLIENT, collection_name=COLLECTION_NAME, embeder=EMBEDER, chunk_limit=3):
    question = question
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[embedd(question, embeder)],
        limit=chunk_limit,
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    # print(json.dumps(retrieved_lines_with_distances, indent=4))
    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )
    user_prompt = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """
    return user_prompt


def retrieve_relevent_chunks_with_ollama(question, milvus_client=MILVUS_CLIENT, collection_name=COLLECTION_NAME, ollama_url=OLLAMA_URL, embeding_model=EMBEDDING_MODEL, chunk_limit=3):
    question = question
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[embed_with_ollama(question, model=embeding_model, ollama_url=ollama_url, timeout=30)],
        limit=chunk_limit,
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    # print(json.dumps(retrieved_lines_with_distances, indent=4))
    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )
    
    user_prompt = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """
    return user_prompt

    





