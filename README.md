# crs2
My personal attempt at constructing a cross-platform, performance pathtracer using C/C++ and NVidia's CUDA platform. 
Code written and tested with NVidia Nsight (mac) and Microsoft VS2015 (win).

**References**

Code is partly based on the "Raytracing in one weekend" mini-books by Peter Shirley:
- http://psgraphics.blogspot.com/
- https://www.amazon.com/Ray-Tracing-Weekend-Minibooks-Book-ebook/dp/B01B5AODD8

And - of course - the industry standard, PBRT v3! http://www.pbrt.org/

Another source of inspiration are the amazing explanations on scratchapixel:
https://www.scratchapixel.com/

<hr>
**Dependencies**
- Nvidia Cuda: https://developer.nvidia.com/cuda-toolkit
- The GLM mathematics library: http://glm.g-truc.net/0.9.8/index.html
- Rapidjson for scene parsing: http://rapidjson.org/

<hr>
<img src="https://github.com/SyntheticPixel/crs2/blob/master/images/20170228-002.jpg" alt="render">

This image (1024x512 pixels, 2000 samples with 8 bounces) was rendered :
- on a GeForce GTX675MX (OSX 10.12.3, 64 bit): 44.57 seconds
- on a GeForce GTX970M (Win10, 64 bit): 28,27 seconds
