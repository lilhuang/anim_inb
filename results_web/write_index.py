import os
import re

import pdb



def write_individual_page(path, model, source, ib, root, folder_name):
	print("confidence confidence")
	if not os.path.exists(os.path.join("./", model, folder_name)):
		os.makedirs(os.path.join("./", model, folder_name))
	targetpath = os.path.join("./", model, folder_name, "index.html")
	file = open(targetpath, "w")
	file.write("<html>\n")
	file.write("<head>\n<title>"+source+" Results: "+source+" "+model+" "+str(ib)+"ib "+"</title>\n")
	file.write("<meta charset=\"utf-8\">\n")
	file.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
	file.write("<link rel=\"stylesheet\" href=\"../../style.css\">\n")
	file.write("</head>\n")
	file.write("<h1>"+source+" Results: "+source+" "+model+" "+str(ib)+"ib</h1>\n")
	file.write("<body>\n")

	file.write("<div class=\"row\">\n")
	file.write("<div class=\"column\">\n")
	file.write("<h4>Estimate</h4>\n")
	file.write("</div>\n")
	file.write("<div class=\"column\">\n")
	file.write("<h4>Ground truth</h4>\n")
	file.write("</div>\n")
	file.write("</div>\n")

	all_examples = os.listdir(path)
	for example in all_examples:
		file.write("<div class=\"row\">\n")

		file.write("<div class=\"column\">\n")
		newroot = os.path.join(path, example)
		frame0_path = os.path.join(newroot, "0.png")
		frame1_path = os.path.join(newroot, "2.png")
		est_ib_path = os.path.join(newroot, "1_est.png")

		bucketpath_0 = os.path.join("../../", "outputs", folder_name, example, "0.png")
		bucketpath_ib = os.path.join("../../", "outputs", folder_name, example, "1_est.png")
		bucketpath_1 = os.path.join("../../", "outputs", folder_name, example, "2.png")

		file.write("<img src=\""+bucketpath_0+"\">\n")
		file.write("<img src=\""+bucketpath_ib+"\">\n")
		file.write("<img src=\""+bucketpath_1+"\">\n")
		file.write("</div>\n")

		file.write("<div class=\"column\">\n")
		gt_ib_path = os.path.join(newroot, "1_gt.png")

		gt_bucketpath_ib = os.path.join("../../", "outputs", folder_name, example, "1_gt.png")

		file.write("<img src=\""+bucketpath_0+"\">\n")
		file.write("<img src=\""+gt_bucketpath_ib+"\">\n")
		file.write("<img src=\""+bucketpath_1+"\">\n")
		file.write("</div>\n")

		file.write("</div>\n")

	file.write("</body>\n</html>\n")

	file.close()

def write_individual_page_by_split_big_blender(path, model, sp, root, folder_name):
	print("jimin u nice keep going")
	if not os.path.exists(os.path.join("./", model, sp)):
		os.makedirs(os.path.join("./", model, sp))
	targetpath = os.path.join("./", model, sp, "index.html")
	file = open(targetpath, "w")
	file.write("<html>\n")
	file.write("<head>\n<title>Big Blender Cube Results: "+sp+" finetune tvl1 7ib</title>\n")
	file.write("<meta charset=\"utf-8\">\n")
	file.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
	file.write("<link rel=\"stylesheet\" href=\"../../style.css\">\n")
	file.write("</head>\n")
	file.write("<h1>Big Blender Cube Results: "+sp+" finetune tvl1 7ib</h1>\n")
	file.write("<body>\n")

	file.write("<div class=\"row\">\n")
	file.write("<div class=\"column\">\n")
	file.write("<h4>Estimate</h4>\n")
	file.write("</div>\n")
	file.write("<div class=\"column\">\n")
	file.write("<h4>Ground truth</h4>\n")
	file.write("</div>\n")
	file.write("</div>\n")

	all_examples = os.listdir(path)
	# pdb.set_trace()
	for example in all_examples:
		file.write("<div class=\"row\">\n")

		file.write("<div class=\"column\">\n")
		newroot = os.path.join(path, example)
		frame0_path = os.path.join(newroot, "0.png")
		frame1_path = os.path.join(newroot, "2.png")
		est_ib_path = os.path.join(newroot, "1_est.png")

		bucketpath_0 = os.path.join("../../", "outputs", folder_name, example, "0.png")
		bucketpath_ib = os.path.join("../../", "outputs", folder_name, example, "1_est.png")
		bucketpath_1 = os.path.join("../../", "outputs", folder_name, example, "2.png")

		file.write("<img src=\""+bucketpath_0+"\">\n")
		file.write("<img src=\""+bucketpath_ib+"\">\n")
		file.write("<img src=\""+bucketpath_1+"\">\n")
		file.write("</div>\n")

		file.write("<div class=\"column\">\n")
		gt_ib_path = os.path.join(newroot, "1_gt.png")

		gt_bucketpath_ib = os.path.join("../../", "outputs", folder_name, example, "1_gt.png")

		file.write("<img src=\""+bucketpath_0+"\">\n")
		file.write("<img src=\""+gt_bucketpath_ib+"\">\n")
		file.write("<img src=\""+bucketpath_1+"\">\n")
		file.write("</div>\n")

		file.write("</div>\n")
	
	file.write("</body>\n</html>\n")

	file.close()




def get_source_links(model, sources, root):
	nextfilename = os.path.join("./", model, "index.html")
	nextfile = open(nextfilename, "w")
	nextfile.write("<html>\n")
	nextfile.write("<head>\n<title>AnimeInterp results</title>\n")
	nextfile.write("<meta charset=\"utf-8\">\n")
	nextfile.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
	nextfile.write("<link rel=\"stylesheet\" href=\"style.css\">\n")
	nextfile.write("</head>\n")
	nextfile.write("<body>\n")
	nextfile.write("<h1>Results by dataset</h1>\n")

	num_ib = [1, 7]
	for source in sources:
		for ib in num_ib:
			folder_name = "avi_"+source+"_"+str(ib)+"ib_"+model+"_results"
			path_maybe = os.path.join(root, "outputs_old", folder_name)
			if not os.path.exists(path_maybe):
				continue
			nextfile.write("<p>\n")
			nextfile.write("<a href=\""+folder_name+"/index.html\" target=\"_blank\">"+folder_name+"</a>\n")
			nextfile.write("</p>\n")
			write_individual_page(path_maybe, model, source, ib, root, folder_name)

	nextfile.write("</body>\n")
	nextfile.write("</html>\n")
	nextfile.close()


def get_source_links_by_split_big_blender(split, model, root):
	path_blah = os.path.join("./", model)
	if not os.path.exists(path_blah):
		os.makedirs(path_blah)
	nextfilename = os.path.join(path_blah, "index.html")
	nextfile = open(nextfilename, "w")
	nextfile.write("<html>\n")
	nextfile.write("<head>\n<title>AnimeInterp results</title>\n")
	nextfile.write("<meta charset=\"utf-8\">\n")
	nextfile.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
	nextfile.write("<link rel=\"stylesheet\" href=\"style.css\">\n")
	nextfile.write("</head>\n")
	nextfile.write("<body>\n")
	nextfile.write("<h1>Results by split on big blender cube data with tvl1 flow</h1>\n")
	for sp in split:
		folder_name = "avi_Blender_7ib_tvl1_finetune_"+sp+"_results"
		path_maybe = os.path.join(root, "outputs", folder_name)
		if not os.path.exists(path_maybe):
			continue
		nextfile.write("<p>\n")
		nextfile.write("<a href=\""+sp+"/index.html\" target=\"_blank\">"+sp+"</a>\n")
		nextfile.write("</p>\n")
		write_individual_page_by_split_big_blender(path_maybe, model, sp, root, folder_name)
	
	nextfile.write("</body>\n")
	nextfile.write("</html>\n")
	nextfile.close()



def main():
	# the old baseline data, fine-tuned on one Blender video
	root = "/vulcanscratch/lilhuang/AnimeInterp"
	models = ["test", "finetuned"]
	sources = ["SU", "Blender"]


	indexfile = open("index.html", "w")
	indexfile.write("<html>\n")
	indexfile.write("<head>\n<title>AnimeInterp results</title>\n")
	indexfile.write("<meta charset=\"utf-8\">\n")
	indexfile.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
	indexfile.write("<link rel=\"stylesheet\" href=\"style.css\">\n")
	indexfile.write("</head>\n")
	indexfile.write("<body>\n")
	indexfile.write("<h1>AnimeInterp results by model</h1>\n")
	for model in models:
		targetpath = os.path.join("./", model, "index.html")
		indexfile.write("<p>\n")
		indexfile.write("<a href=\""+targetpath+"\" target=\"_blank\">"+model+"</a>\n")
		indexfile.write("</p>\n")
		get_source_links(model, sources, root)

	#new baseline data, fine-tuned on big Blender data
	models_big_blender = ["big_blender_finetuned"]
	data_split = ["TRAIN", "TEST"]
	for model in models_big_blender:
		targetpath = os.path.join("./", model, "index.html")
		indexfile.write("<p>\n")
		indexfile.write("<a href=\""+targetpath+"\" target=\"_blank\">"+model+"</a>\n")
		indexfile.write("</p>\n")
		get_source_links_by_split_big_blender(data_split, model, root)


	indexfile.write("</body>\n</html>\n")

	indexfile.close()
	



if __name__ == '__main__':
	main()



