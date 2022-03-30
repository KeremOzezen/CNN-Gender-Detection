# Bu dosya programın başlangıcını içerir

# Eğitilmiş bir VGG modeli var mı diye kontrol eder.
# Eğer bulamazsa egitim.py dosyasını çağırarak network modelini oluşturur
# Eğer model mevcut ise test.py dosyasını çağırarak networkü test eder ve sonuçları görüntüler

import os
import os.path

def main():
    if not os.path.isfile( "vgg.model" ):
        os.system( "egitim.py" )
    
    os.system( "test.py" )


if __name__ == "__main__":
    main()
