from .base import BasePreprocessor
from google import genai
import pdfplumber
import os
import time
import random
from google.genai import errors

MODEL = "gemini-2.5-flash-lite"

class PaperLLMProcessor(BasePreprocessor):
    POLL_INTERVAL = 30  # seconds between status polls

    def __init__(self):
        super().__init__()
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    @property
    def name(self) -> str:
        return "paper_llm_processor"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process_document(self, source_path: str) -> str:
        tables, contexts = self._extract_interleaved_content(source_path)

        if not tables:
            return contexts[0] if contexts else ""

        # Build one prompt per table and submit them all in a single batch job
        descriptions = self._batch_process_tables(tables)

        # Reassemble the document: interleave context blocks with table descriptions
        text = ""
        for i, description in enumerate(descriptions):
            text += contexts[i] + "\n"
            text += description + "\n"
        text += contexts[-1]
        return text

    # ------------------------------------------------------------------
    # Batch API helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, table: dict) -> str:
        return f"""
            Du hast nun eine Aufgabe zu erledigen. Aufgabenbeschreibung:
Du erhältst eine Tabelle (im 2D-Array-Format mit einer Beschriftung). 
Du musst eine natürlichsprachliche Beschreibung des Inhalts der 
Tabelle erstellen. Du darfst ausschließlich Inhalte basierend auf 
dem Tabelleninhalt generieren; erzeuge keine anderen verwandten 
oder unzusammenhängenden Inhalte. Hier ist ein Beispiel.
Beschriftung: Parameter für ip link add name und ip link del dev.
Tabelle:
[['Parameter', 'Beschreibung', 'Wert'], ['name NAME', 'Gibt den Namen 
einer Bridge an.', 'Der Wert ist eine Zeichenfolge von 1 bis 15 Zeichen 
unter Beachtung der Groß- und Kleinschreibung ohne Leerzeichen.'], 
['dev DEV', 'Gibt den Namen einer Bridge an.', 'Der Wert ist eine 
Zeichenfolge von 1 bis 15 Zeichen unter Beachtung der Groß- und 
Kleinschreibung ohne Leerzeichen.'], ['type bridge', 'Gibt an, dass der 
Gerätetyp eine Bridge ist.', '-']].
Beschreibung: Die Tabelle liefert Details zu den Parametern für die 
Befehle ip link add name und ip link del dev. Es gibt verschiedene 
Parameter für die Konfiguration einer Bridge. Der Parameter "name NAME" 
dient zur Angabe des Namens einer Bridge und akzeptiert eine Zeichenfolge 
von 1 bis 15 Zeichen ohne Leerzeichen unter Beachtung der Groß- und 
Kleinschreibung. Der Parameter "dev DEV" gibt ebenfalls den Namen einer 
Bridge an und erfordert eine Zeichenfolge von 1 bis 15 Zeichen ohne 
Leerzeichen unter Beachtung der Groß- und Kleinschreibung. Der Parameter 
"type bridge" gibt an, dass das zu konfigurierende Gerät vom Typ 'bridge' 
ist. Er erfordert keinen spezifischen Wert.
Beschriftung: {table['caption']}
Tabelle: {table['data']}
Beschreibung:
        """

    def _batch_process_tables(self, tables: list) -> list:
        """Submit all table prompts as one Gemini Batch job and return ordered descriptions."""
        from google.genai.types import JobState

        inline_requests = [
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": self._build_prompt(t)}],
                    }
                ]
            }
            for t in tables
        ]

        print(f"Submitting batch job with {len(inline_requests)} request(s)...")
        job = self.client.batches.create(
            model=self.MODEL,
            src=inline_requests,
            config={"display_name": "paper-llm-processor-tables"},
        )
        print(f"Batch job created: {job.name}")

        # Poll until the job reaches a terminal state
        terminal_states = {
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
            JobState.JOB_STATE_PAUSED,
        }

        while job.state not in terminal_states:
            print(f"  Job state: {job.state}  — waiting {self.POLL_INTERVAL}s...")
            time.sleep(self.POLL_INTERVAL)
            job = self.client.batches.get(name=job.name)

        print(f"Batch job finished with state: {job.state}")

        if job.state != JobState.JOB_STATE_SUCCEEDED:
            raise RuntimeError(
                f"Batch job {job.name} did not succeed (state={job.state}). "
                "Check the Google Cloud console for details."
            )

        # Access the responses through the `dest` attribute
        responses = job.dest.inlined_responses  
        descriptions = []
        
        for i, resp in enumerate(responses):
            # Good practice: Check if this specific inline request failed
            if resp.error:
                print(f"  Warning: Request {i} failed with error: {resp.error}")
                text = ""
            elif resp.response:
                try:
                    # Use the SDK's built-in .text shortcut for cleaner extraction
                    text = resp.response.text.strip()
                except AttributeError as exc:
                    print(f"  Warning: could not extract text for request {i}: {exc}")
                    text = ""
            else:
                text = ""
                
            descriptions.append(text)

        return descriptions

    def _extract_interleaved_content(self, pdf_path: str):
        """
        Extracts text and tables chronologically to allow for perfect document reconstruction.
        
        Returns:
            tuple: (document_tables, contexts)
                - document_tables: List of N table dictionaries.
                - contexts: List of N+1 strings representing the text between tables.
                Format: [Text Before T1, Text After T1, Text After T2, ..., Text After TN]
        """
        document_tables = []
        contexts = []
        
        # Acts as a running buffer for the text between tables
        current_context_buffer = [] 
        last_table_state = None 

        # Helper function to safely crop and extract text
        def extract_slice(page, top, bottom):
            top, bottom = max(0, top), min(page.height, bottom)
            if top >= bottom:
                return ""
            text = page.crop((0, top, page.width, bottom)).extract_text()
            return text.strip() if text else ""

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                
                tables = page.find_tables()
                # Sort vertically to ensure chronological processing
                tables.sort(key=lambda t: t.bbox[1]) 
                
                cursor_y = 0 # Tracks our vertical position on the page
                
                for table_idx, table in enumerate(tables, start=1):
                    x0, top, x1, bottom = table.bbox
                    table_data = table.extract()
                    
                    if not table_data or not table_data[0]:
                        continue
                    
                    col_count = len(table_data[0])
                    is_continuation = False
                    
                    # --- Continuation Logic ---
                    if last_table_state and table_idx == 1:
                        is_next_page = (page_num == last_table_state["page"] + 1)
                        same_columns = (col_count == last_table_state["col_count"])
                        prev_near_bottom = last_table_state["bottom"] > (page.height * 0.75)
                        curr_near_top = top < (page.height * 0.25)
                        
                        if is_next_page and same_columns and prev_near_bottom and curr_near_top:
                            is_continuation = True

                    if is_continuation:
                        # Extract any text *above* the continuation (like a page header)
                        text_above = extract_slice(page, cursor_y, top)
                        if text_above:
                            current_context_buffer.append(text_above)
                        
                        # Merge table data
                        if table_data[0] == last_table_state["header_row"]:
                            last_table_state["data_ref"].extend(table_data[1:])
                        else:
                            last_table_state["data_ref"].extend(table_data)
                        
                        last_table_state["bottom"] = bottom
                        last_table_state["page"] = page_num
                        last_table_state["dict_ref"]["end_page"] = page_num
                        
                        cursor_y = bottom
                        
                    else:
                        # --- NEW TABLE DETECTED ---
                        # 1. Determine where the caption starts
                        caption_top = max(cursor_y, top - 50)
                        
                        # 2. Extract the text block BEFORE the caption/table
                        text_before = extract_slice(page, cursor_y, caption_top)
                        if text_before:
                            current_context_buffer.append(text_before)
                            
                        # 3. WE HIT A TABLE: Close the current context and save it
                        contexts.append("\n".join(current_context_buffer).strip())
                        current_context_buffer = [] # Reset buffer for the next context block
                        
                        # 4. Extract Caption
                        caption_text = extract_slice(page, caption_top, top)
                        
                        # 5. Save the Table
                        new_table_dict = {
                            "start_page": page_num,
                            "end_page": page_num,
                            "table_index": len(document_tables) + 1,
                            "caption": caption_text if caption_text else None,
                            "data": table_data
                        }
                        document_tables.append(new_table_dict)
                        
                        # 6. Set Trackers
                        last_table_state = {
                            "page": page_num,
                            "bottom": bottom,
                            "col_count": col_count,
                            "header_row": table_data[0],
                            "data_ref": new_table_dict["data"],
                            "dict_ref": new_table_dict
                        }
                        
                        cursor_y = bottom # Move cursor past the table
                
                # --- End of Page: Extract remaining text ---
                text_below = extract_slice(page, cursor_y, page.height)
                if text_below:
                    current_context_buffer.append(text_below)

        # --- End of Document: Close the final context block ---
        contexts.append("\n".join(current_context_buffer).strip())
        
        return document_tables, contexts

class DirectLLMProcessor(BasePreprocessor):
    """
    Uses LLM to convert markdown to text.
    Requires a Markdown processor to be run before (or the markdown documents as input).
    """
    def __init__(self):
        super().__init__()
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    @property
    def name(self) -> str:
        return "direct_llm"

    def _build_prompt(self, text: str) -> str:
        return f"""
            Wandle das folgende Markdown-Dokument in klaren, informationsdichten Fließtext um, der für eine semantische Suchmaschine (RAG-Pipeline) optimiert ist.

            Befolge dabei strikt diese Regeln:
            1. Informationserhalt: Übernimm ausnahmslos jede Information, jede Zahl und jedes Detail.
            2. Hohe Entitätsdichte (WICHTIG): Vermeide Pronomen (er, sie, es, diese) über Absatzgrenzen hinweg. Wiederhole stattdessen immer wieder die konkreten Eigennamen, Produktnamen oder Fachbegriffe, damit jeder Absatz auch isoliert verständlich bleibt.
            3. Tabellen & Listen: Wandle Tabellen und Aufzählungen in zusammenhängende Textabsätze um. Formuliere die Beziehungen zwischen Spalten/Zeilen in ganzen Sätzen aus.
            4. Überschriften als Kontext: Nutze Überschriften, um den darauffolgenden Absatz mit einem klaren Kontext-Satz zu beginnen (z. B. statt nur "Spezifikationen:" -> "Die technischen Spezifikationen des Geräts X umfassen...").
            5. Formatierung: Entferne alle Markdown-Zeichen (*, #, -, |). Behalte klare Absatzumbrüche bei, um logische Trennungen beizubehalten.

            Input Text:
            ---
            {text}
            ---
            
        """

    def process_document(self, source_path: str) -> str:
        with open(source_path, 'r') as f:
            text = f.read()
            
        print(text)
        
        # Build the prompt once outside the loop to save processing time
        prompt_contents = self._build_prompt(text)
        
        max_retries = 5
        base_delay = 2.0  # Starts with a 2-second delay

        for attempt in range(max_retries):
            try:
                interaction = self.client.interactions.create(
                    model=MODEL,
                    input=prompt_contents,
                    service_tier='flex'
                )
                return interaction.output_text
                                
            except (errors.ServerError, errors.APIError) as e:
                # Safely extract the status code (503 for ServerError, 429 for APIError)
                status_code = getattr(e, 'code', None)
                
                # Check if the error is a temporary bottleneck we can wait out
                if status_code in [503, 429] or "503" in str(e):
                    if attempt < max_retries - 1:
                        # Exponential backoff: 2s, 4s, 8s, 16s + 1-3 seconds of random jitter
                        sleep_time = (base_delay * (2 ** attempt)) + random.uniform(1.0, 3.0)
                        print(f"[API {status_code}] Backend stalled. Retrying in {sleep_time:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(sleep_time)
                    else:
                        raise RuntimeError(f"Failed to process document after {max_retries} attempts due to API limits. Last error: {e}")
                else:
                    # If it's a 400 Bad Request (e.g., token limit exceeded, bad JSON), 
                    # retrying won't fix it. Raise the error immediately.
                    raise e



# tables, text_blocks = _extract_interleaved_content("documents/MHB.pdf")

# print(tables[2]['caption'])
# print(tables[2]['data'])
# print(text_blocks)
