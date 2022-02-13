import insightface_paddle as face
import time
import cv2

# parser = face.parser()
# args = parser.parse_args()
# start_time = time.time()
# args.print_info=False
# args.rec = True
# args.index = "./index.bin"
# input_path = "./hien.jpg"

# predictor = face.InsightFace(args)
# res = predictor.predict(input_path,print_info=True)
# print(res)

# end_time = time.time()
# #tính thời gian chạy của thuật toán Python
# elapsed_time = end_time - start_time
# print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")



parser = face.parser()
args = parser.parse_args()
args.rec = True
args.det = True
args.index = "./index.bin"
path = "./hien.jpg"
img = cv2.imread(path)[:, :, ::-1]

predictor = face.InsightFace(args)
res = predictor.predict(img, print_info=True)
print(res)