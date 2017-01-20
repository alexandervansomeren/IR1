import pyndri
import utils

def main():
    index = pyndri.Index('index/')
    print("There are %d documents in this collection." % (index.maximum_document() - index.document_base()))




if __name__ == "__main__":
    main()




