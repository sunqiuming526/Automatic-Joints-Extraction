# Automatic-Joints-Extraction

Referenced Paper: Pose-independent joint extraction from scanned human body

一种姿态无关的人体模型骨骼提取方法

https://www.researchgate.net/publication/290562205_Pose-independent_joint_extraction_from_scanned_human_body

TODO [2/7]
- [x] Find 5 characteristic points. (Head, Left hand, Right hand, Left foot, Right foot)
- [x] Calculate the Morse function. (In this project, I set the Morse value as the geodesic distance of any point to source point which is the head point in this case)
- [ ] Draw the Level-Set-Curve.
- [ ] Calculate the centroid of each Level-Set-Curve to extract the skeleton.
- [ ] Identify the semantic of the joints.
- [ ] Get the joint position.
