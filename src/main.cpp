#include <time.h> 
#include <memory>
#include <string>
#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>

int main()
{
	std::string modle_dir = "../model/resnet18.pt";
	std::string image_dir = "../data/dog.png";
	std::string label_file = "../data/synset_words.txt";

	// load model
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(modle_dir);
	module->to(at::kCUDA);

	if (module == nullptr)
	{
		std::cout << "load error" << std::endl;
		return -1;
	}
	else
		std::cout << "load success" << std::endl;

	// read input image
	cv::Mat image = cv::imread(image_dir, 1);
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	cv::resize(image, image, cv::Size(224, 224));
	
	// convert opencv Mat to torch Tensor
	torch::Tensor tensor_image = torch::from_blob(image.data, { 1,224, 224,3 }, torch::kByte);
	tensor_image = tensor_image.permute({ 0,3,1,2 });
	tensor_image = tensor_image.toType(torch::kFloat);

	// sub mean and div std
	double mean[] = { 0.485, 0.456, 0.406 };
	double std[] = {0.229, 0.224, 0.225};

	tensor_image = tensor_image.div(255);
	tensor_image[0][0] = tensor_image[0][0].sub_(mean[0]).div_(std[0]);
	tensor_image[0][1] = tensor_image[0][1].sub_(mean[1]).div_(std[1]);
	tensor_image[0][2] = tensor_image[0][2].sub_(mean[2]).div_(std[2]);

	tensor_image = tensor_image.to(torch::kCUDA);

	// Execute the model and turn its output into a tensor.
	auto start = clock();
	torch::Tensor output = module->forward({tensor_image}).toTensor();
	auto end = clock();
	std::cout << "time: " << static_cast<double>(end - start) / CLOCKS_PER_SEC << 's' << std::endl;
	output = output.to(torch::kCPU);

	// Load labels
	std::ifstream rf(label_file.c_str());
	std::string line;
	std::vector<std::string> labels;
	while (std::getline(rf, line))
		labels.push_back(line);

	// print predicted top-5 labels
	std::tuple<torch::Tensor, torch::Tensor> result = output.sort(-1, true);
	torch::Tensor top_scores = std::get<0>(result)[0];
	torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);

	auto top_scores_a = top_scores.accessor<float, 1>();
	auto top_idxs_a = top_idxs.accessor<int, 1>();

	for (int i = 0; i < 5; ++i) 
	{
		int idx = top_idxs_a[i];
		std::cout << "top-" << i + 1 << " label: ";
		std::cout << labels[idx] << ", score: " << top_scores_a[i] << std::endl;
	}

	system("pause");
	return 0;
}

// top - 1 label: n02108422 bull mastiff, score : 17.9906
// top - 2 label : n02093428 American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier, score : 13.3816
// top - 3 label : n02109047 Great Dane, score : 12.8467
// top - 4 label : n02093256 Staffordshire bullterrier, Staffordshire bull terrier, score : 12.1757
// top - 5 label : n02110958 pug, pug - dog, score : 11.9858
// time: 0.85s