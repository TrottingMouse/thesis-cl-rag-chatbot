import pdfplumber


# use this plus prompt from paper for llm conversion? test method and direct gemini?
def extract_raw_pdf_tables(pdf_path, page_num):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        
        # 1. Extract the title/caption (assuming it's right above the table)
        # Note: In production, you'd look at text blocks above the table bounding boxes.
        caption = "Table Caption: Parameters for the system configuration." 
        
        # 2. Extract the structural table objects
        tables = page.find_tables()
        if not tables:
            return None, None
            
        # Get the first table on the page
        target_table = tables[0]
        
        # Get the strict matrix shape (N x M grid layout)
        grid_data = target_table.extract() 
        
        return caption, grid_data

caption, grid = extract_raw_pdf_tables("documents/PO.pdf", 0)
with open("grid.txt", "w") as f:
    f.write(str(grid))

"""Now you have a task to complete. Task description:
You will be given a table (with the 2d array format
with the Caption). You need to generate a natural
language description of the contents of the table. You
can only generate content from the table content, do
not generate other related or unrelated content. Here
is an examples.
Table: Caption: Parameters for the ip link add name
and ip link del dev.
[[’Parameter’, ’Description’, ’Value’], [’name
NAME’, ’Specifies the name of a bridge.’, ’The value
is a string of 1 to 15 case-sensitive characters with-
out spaces.’], [’dev DEV’, ’Specifies the name of
a bridge.’, ’The value is a string of 1 to 15 case-
sensitive characters without spaces.’], [’type bridge’,
’Indicates that the device type is bridge.’, ’-’]].
Description: The table provides details on the
parameters for the ip link add name and ip link del
dev commands. There are different parameters for
configuring a bridge. The "name NAME" parameter
is for specifying the name of a bridge and accepts
a string with 1 to 15 case-sensitive characters,
excluding spaces. The "dev DEV" parameter also
specifies the name of a bridge and requires a string
of 1 to 15 case-sensitive characters without spaces.
The "type bridge" parameter indicates that the device
being configured is of the type ’bridge.’ It does not
require a specific value.
Table: {Table}
Description:"""