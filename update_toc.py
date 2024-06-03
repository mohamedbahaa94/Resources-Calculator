import sys
import win32com.client as win32
import pythoncom

def update_toc(file_path):
    pythoncom.CoInitialize()
    word_app = win32.gencache.EnsureDispatch('Word.Application')
    word_app.Visible = False
    try:
        doc = word_app.Documents.Open(file_path)
        doc.TablesOfContents(1).Update()
        doc.Close(SaveChanges=True)
    except Exception as e:
        print(f"Failed to update TOC: {str(e)}", file=sys.stderr)
    finally:
        word_app.Quit()
        pythoncom.CoUninitialize()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        update_toc(file_path)
    else:
        print("Usage: python update_toc.py <file_path>")
