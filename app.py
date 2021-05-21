import os
from flask import Flask,render_template,request,flash
import torch
import torch.optim
import io
import torchvision.transforms as transforms
from PIL import Image
from werkzeug.utils import secure_filename
from torchvision.utils import save_image
from networks import Generator

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(12).hex()
#@app.route('/')
#def main():    
#    return (render_template('main.html'))

@app.route('/',methods =['POST','GET'])
def predict():
    if request.method =='POST':
        if 'file' not in request.files:
            flash('No file')
            return (render_template('main.html'))
        file = request.files['file']
        if file=='':
            flash('No file')
            return (render_template('main.html'))
         
        
        for f in os.listdir('static'):
            os.remove(os.path.join('static', f))
            
        
        
        filename = secure_filename(file.filename)
        
        
        img = file.read()
        result = enhanceImage(img)        
        if result['status']==200:
            
            image = Image.open(io.BytesIO(img))
            image = image.resize(result["shape"])
            
            image.save(f"static/{filename}")
            return render_template("main.html",user_image=f'static/{filename}',enhanced_image=f"static/{result['filename']}")
        else:
            
            flash('Oops something went wrong',result['message'])
            return (render_template('main.html'))
    return (render_template('main.html'))

if __name__=="__main__":
    app.run()


def transform_image(img,img_height,img_width):
    preprocess = transforms.Compose([transforms.Resize((img_height,img_width)),                                    
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], )
                                    
                                        ])
    image = Image.open(io.BytesIO(img))
    return preprocess(image).unsqueeze(0)


def enhanceImage(img):
    try:
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        img_height,img_width = 256,256
        
    
        tensor = transform_image(img,img_height,img_width)
        generatoru_path = os.path.join('trained_weights','genu.pth.tar')
        modelU = torch.load(generatoru_path,map_location=device)
        generatorD = Generator().to(device)
        generatorU = Generator().to(device)
        
        optimizer_G = torch.optim.Adam(list(generatorD.parameters())+list(generatorU.parameters()), lr=2e-4, betas=(0.5, 0.999))
        
        generatorU.load_state_dict(modelU['state_dict'])
        optimizer_G.load_state_dict(modelU['optimizer'])
            
        generatorU.eval()
        enhanced_img = generatorU(tensor)
        filename = os.urandom(12).hex()+".png"
        save_image(enhanced_img*0.5+0.5,f"static/{filename}")
        return {"status":200,"filename":filename,"shape":enhanced_img.size()[2:]}
    
    except Exception as e:
        return {"status":400,"message":str(e)}
  
#
#            print(filename)
#            with open(f'static/{filename}','wb')  as f: f.write(img)
#            print(os.listdir('static'))
#            print(result['filename'])