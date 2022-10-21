1. AlphaStar Unplugged https://github.com/deepmind/alphastar/tree/main/alphastar/unplugged/data
这个是他们处理starcraft 的replay数据的pipeline，他们是把第三方的数据统一处理之后然后在使用。我觉得如果cross domain的话可能统一一下数据的格式会比较好一点。
2. https://github.com/deepmind/envlogger 这个是他们的用来采集数据时的一个库，生成出来的也是同意文件的格式，不过内部外部用的格式可能稍微不一样， 但是原理差不多。
https://github.com/google-research/rlds 这个是tf 下面的一个google采用的统一格式，我印象是deepmind自己的格式跟这个有点出入，但是也是统一的。内部用的应该是类似于SSTable的一个格式 https://www.igvita.com/2012/02/06/sstable-and-log-structured-storage-leveldb/ ，这个格式对于数据大规模的的IO比较友好， 我没有更多的细节了。
tldr，那天也跟朋友正好聊到了，deepmind算的快其实就是卡多+infra好，本身gato从设计方面没有什么太多的秘密，大部分的活其实都在infra层面处理掉了。
3. 训练的话用到了内部的一个部署到云的库，https://github.com/deepmind/xmanager/tree/main/xmanager 这个是他们开源的部分。大概就是一个experiment开一个training的分布式job再开一些evaluator （分布式） job，这个框架会自动把这些job在borg上面调度起来。
