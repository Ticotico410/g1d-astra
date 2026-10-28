## 查看摄像头设备号
```bash
#从里面找到每个摄像头开始有差异的字段
ls -lt /sys/class/video4linux/

(base) unitree@ubuntu:~$ ls -lt /sys/class/video4linux/
total 0
lrwxrwxrwx 1 root root 0 Sep 16 15:25 video0 -> ../../devices/platform/3610000.xhci/usb1/1-2/1-2.1/1-2.1:1.0/video4linux/video0
lrwxrwxrwx 1 root root 0 Sep 16 15:25 video1 -> ../../devices/platform/3610000.xhci/usb1/1-2/1-2.1/1-2.1:1.0/video4linux/video1
lrwxrwxrwx 1 root root 0 Sep 16 15:25 video2 -> ../../devices/platform/3610000.xhci/usb1/1-3/1-3.1/1-3.1:1.0/video4linux/video2
lrwxrwxrwx 1 root root 0 Sep 16 15:25 video3 -> ../../devices/platform/3610000.xhci/usb1/1-3/1-3.1/1-3.1:1.0/video4linux/video3
lrwxrwxrwx 1 root root 0 Sep 16 15:25 video4 -> ../../devices/platform/3610000.xhci/usb1/1-3/1-3.2/1-3.2:1.0/video4linux/video4
lrwxrwxrwx 1 root root 0 Sep 16 15:25 video5 -> ../../devices/platform/3610000.xhci/usb1/1-3/1-3.2/1-3.2:1.0/video4linux/video5

例如上面这样的就是1-2.1、1-3.2、1-3.1
```


## 新建udev规则
```bash
sudo vi /etc/udev/rules.d/99-usb-cameras.rules
```

## 把下面的内容放进去
```bash
# head
SUBSYSTEM=="video4linux", KERNELS=="1-2.1", ATTR{index}=="0", SYMLINK+="cam_head"

# wrist left
SUBSYSTEM=="video4linux", KERNELS=="1-3.2", ATTR{index}=="0", SYMLINK+="cam_wrist_left"

# wrist right
SUBSYSTEM=="video4linux", KERNELS=="1-3.1", ATTR{index}=="0", SYMLINK+="cam_wrist_right"

```

## 生效udev规则
```bash
sudo udevadm control --reload-rules && sudo udevadm trigger && ls -la /dev/cam_*
```

## 生效自启动
```bash
#把 image_server 文件夹 放到 /home/unitree 下
#把 image-server.service 放到 /etc/systemd/system/ 下
sudo systemctl daemon-reload
sudo systemctl enable image-server.service  
sudo systemctl start  image-server.service   
systemctl status image-server
```

