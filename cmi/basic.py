import sys

Adam_info = {"height": 175, "weight": 60}
Eve_info = [165, 50]  # 两个数字分别代表夏娃的身高和体重

if Eve_info[1] > Adam_info["weight"]:
    print("很遗憾，夏娃的体重现在是" + str(Eve_info[1]) + "公斤。")
elif Eve_info[1] == Adam_info["weight"]:
    print("很遗憾，夏娃的体重和亚当一样。")
    sys.exit(0)
else:
    print("重要的事儿说3遍!")
    for i in range(3):
        print("夏娃没有亚当重，她的体重只有" + str(Eve_info[1]) + "公斤。")
