#
# make lmdb database given siamese pairs and labels
#
import os, re, caffe, lmdb, shutil
import numpy as np


amosImageDir = '/u/eag-d1/scratch/ted/deeptransient/AMOS_close/'
transientDir = '/u/eag-d1/data/transient/transient/imageAlignedLD/'
image_label_file = '/u/eag-d1/scratch/ted/webcamattri/transient/annotations.csv'

# location to store train/val database
db_location = '/u/eag-d1/scratch/ted/deeptransient/lmdbs/siamese/'

image_size = 256

transientNames = [
  'dirty','daylight','night','sunrisesunset','dawndusk',
  'sunny','clouds','fog','storm','snow',
  'warm','cold','busy','beautiful','flowers',
  'spring','summer','autumn','winter','glowing',
  'colorful','dull','rugged','midday','dark',
  'bright','dry','moist','windy','rain','ice',
  'cluttered','soothing','stressful','exciting',
  'sentimental','mysterious','boring','gloomy','lush'
]
transient2colnum = { name:(2*ix+1) for ix, name in enumerate(transientNames)}

interest_transients = transientNames


# load transient dataset labels
transientLabels = {}
with open(image_label_file, 'r') as f:
  for line in f.readlines():
    cell = re.split('\t|,', line)
    label = [ float(cell[transient2colnum[key]].strip()) for key in transientNames ]
    interest_label = [ float(cell[transient2colnum[key]].strip()) for key in interest_transients ]
    name = transientDir + cell[0].strip()
    transientLabels[name] = {'full':label, 'interest':interest_label}


def parse_pairs_txt(pairs_txt):
  pairs = []
  with open(pairs_txt, 'r') as f:
    for line in f.readlines():
      cell = line.split(' ')
      unit = [cell[0].strip(), cell[1].strip(), int(cell[2].strip())]
      pairs.append(unit)
      
  return pairs


def chunk(s, n):
  assert n > 0
  while len(s) >= n:
    yield s[:n]
    s = s[n:]
  if len(s):
    yield s

  
def make_database(pairs_txt, mode):

  dbDir = os.path.join(db_location, mode)

  if mode == 'debug' and os.path.isdir(dbDir): # debug mode
    shutil.rmtree(dbDir)

  if os.path.isdir(dbDir):
    raise Exception(dbDir + ' already exists. Delete it')

  os.mkdir(dbDir)

  # initiate lmdb env
  im_db = lmdb.Environment(dbDir + '/image_db', map_size=1000000000000)
  lb_db = lmdb.Environment(dbDir + '/label_db', map_size=1000000000000)

  
  pairs = parse_pairs_txt(pairs_txt)
  chunks = chunk(pairs, 1000)

  cnt = 0
  for pair_chunk in chunks:

    with im_db.begin(write=True) as im_db_txn:
      with lb_db.begin(write=True) as lb_db_txn:

        for key, pair in enumerate(pair_chunk):
          
          imgName0 = pair[0]
          imgName1 = pair[1]
          siaLabel = pair[2]
          allLabel = np.asarray(transientLabels[imgName0]['full'])
          traLabel = np.asarray(transientLabels[imgName0]['interest'])
      
          # load image
          try:
            im0 = caffe.io.load_image(imgName0)
            im1 = caffe.io.load_image(imgName1)
          except:
            print 'bad image, skip it.'
            continue
      
          # resizing
          im0 = caffe.io.resize_image(im0, (image_size, image_size))
          im1 = caffe.io.resize_image(im1, (image_size, image_size))
      
          # channel swap for pre-trained (RGB -> BGR)
          im0 = im0[:, :, [2,1,0]]
          im1 = im1[:, :, [2,1,0]]
      
          # concatenate two image in channel axis
          img = np.zeros((image_size, image_size, 6))
          img[:,:,0:3] = im0
          img[:,:,3: ] = im1
      
          # make channels x height x width
          img = img.swapaxes(0,2).swapaxes(1,2)
      
          # convert to uint8
          img = (255*img).astype(np.uint8, copy=False) 
      
          # [images siaLabel] to datum 
          img_datum = caffe.io.array_to_datum(img, siaLabel)
          img_str = img_datum.SerializeToString()
      
      
          # transient label to datum
          traLabel = traLabel.reshape((len(traLabel), 1, 1)) # reshaping to caffe format
          tra_datum = caffe.io.array_to_datum(traLabel)
          tra_datum.ClearField('label')
          tra_str = tra_datum.SerializeToString()
      
          #
          # write datum to lmdb
          #
          key_str = str(key)
          im_db_txn.put(key_str, img_str)
          lb_db_txn.put(key_str, tra_str)

    cnt += len(pair_chunk)
    print "processed %d of %d pairs (%s)" % (cnt, len(pairs), mode)



make_database('siamese_pairs/debug_pairs.txt', 'debug')
make_database('siamese_pairs/train_pairs.txt', 'train')
make_database('siamese_pairs/val_pairs.txt', 'val')
