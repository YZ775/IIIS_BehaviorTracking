# Requirement
## Python3
numpy 

pyserial


# Usage
MacOS
```
$python MACOS_track_color.py
```
Windows
```
$python WINDOWS_track_color.py
```


<br>

```
select Source
0 or 1 : from camera
else: video PATH
>>
```
To use webcamera,input 0 or 1

Windows
```
Select Device
only found COM1
COM1
```
<br>

MacOS
```
Select Device
  0: open /dev/cu.MALS
  1: open /dev/cu.SOC
  2: open /dev/cu.Bluetooth-Incoming-Port
  3: open /dev/cu.usbserial-14530
input number of target port >>
```

If USB serial device is only Arduino, automatically select device. 
Otherwise,input device number(MacOS:/dev/cu.usbserial-14530, Windows: COM*)
<br>
<br>
<br>
<br>


```
Enter motion index coefficient(default:1.0) >>
```
Please input motion index coefficient. 

This coefficient determin how much movement is 5V  

Motion index 100 is 5V. 
<br>
<br>
<br>
<br>
<br>
<br>
<img width="300" alt="スクリーンショット 2019-07-31 15 05 42" src="https://user-images.githubusercontent.com/33110971/62187666-12ecdf00-b3a5-11e9-8b33-036b5868a5e6.png">
 
Picture of webcamera will appear,then left click and right click the mouse. This determin color of mouse. 

Next,Enter "Q"key to start program.
<br>
<br>
Again,Enter "Q"key to stop program.


output movie is  
<br>
MacOS
```
out.mp4
```
Windows
```
out.avi
```


<img width="400" alt="スクリーンショット 2019-07-31 15 05 42" src="https://user-images.githubusercontent.com/33110971/62187856-a3c3ba80-b3a5-11e9-807d-e2383bf17146.gif">

This Number is motion index