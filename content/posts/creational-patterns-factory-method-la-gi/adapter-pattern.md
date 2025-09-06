![banner](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgKb9KVJma41T66g9Zc7sqf2qxk2FmENB_mgRE0KPcVW1ttHUHaGwKEgdY8oU3BqwjIKxCMaqm-Dk_MbyYvVGl4PVVS0gkFiSYVv8vx6-ExryoPy0YYyEuJDkVAhnGbSmxY2XFwxTZVWqwf-EotthbhyCeSxrlAC7j8sQKnzlhCs6ZR4qZh3UFf9hoZQTU/w640-h360/banner.jpg)

Khi áp dụng một design pattern tốt sẽ giúp bạn tiết kiệm thời gian và công sức sau này. Bạn có thể tái sử dụng code, mở rộng khi cần thiết. Bạn có từng nhớ lần đọc code cuối cùng không? Bạn có thấy code của bạn dễ dàng chỉnh sửa mở rộng không? Nếu không, cùng tôi tìm hiểu về design pattern và cách áp dụng nó vào trong dự án của bạn.

Design Pattern được chia ra làm 3 loại chính, bao gồm:

1. **[Creational Patterns](https://www.levandong.dev/search/label/Creational%20Patterns)**: giúp bạn có thể tạo ra object để sử dụng trong ứng dụng. Mục tiêu là giúp code trở nên ít phụ thuộc, không phụ thuộc vào từ khoá `new` quá nhiều.
2. **[Structural Patterns](https://www.levandong.dev/search/label/Structural%20Patterns)**: giúp bạn tổ chức các class và struct sao cho dễ dàng mở rộng mà không ảnh hưởng đến hệ thống.
3. **[Behavioral Patterns](https://www.levandong.dev/search/label/Behavioral%20Patterns)**: giúp bạn quản lý hành vi của class hay struct. Tối ưu hoá mối liên hệ giữa các class bằng cách đưa ra các quy tắc giao tiếp giữa các class.

## Vấn đề

Giả sử bạn mới mua máy ảnh Nikon... *(nội dung tiếp tục)*

## Adapter là gì?

Với cách đặt vấn đề bên trên... *(nội dung tiếp tục)*

## Cấu trúc của Adapter Pattern

Chúng ta hãy thử xem qua sơ đồ cấu trúc của Adapter Pattern:

![Cấu trúc Adapter Pattern](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEh06j7CXBbOm6qmFoWhPPQxt_vTpoq4oat6qTVPGQneJvIk6aSlVjUKbqnjhS0IDeyRXRUpBz61V49PZk2MrV4poVLCznKkBzj_baCqG54phq-wbG4hfJhDGd214Cu3kgvBlIjU_nVvaiYYK2Wfj8LldIHTf7Te9nuU0v0QBy5QwsAzRHfp-rh0azEDNzw/w329-h400/cau-truc-adapter-pattern.png)

**Client**: là class chứa business logic.  
**Target**: là interface...  
**Adapter**: là class kế thừa từ thiết kế của Target.  
**Adaptee**: là class không tương thích...

![nikon adapter](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjVXciUJ5BPcIDklJB46GDh8CPDBpldhcAhMgE8E6smKsJeUWhqnUOAKZTsuZXwEVwW3a_0Tz6PbchQ8qmFJXttfXBFyrrJoivq6gHtgNvMPXqnnEzBXD_0oKHiIHMBRKB1qPHPEv-MbByLeiHLNP3uXLbpieLCgoa65unOGNTP0ycxODeAzTuMAmxwWjU/w640-h538/nikonztof-adapter-pattern.png)

## Cách triển khai Adapter Pattern

### Triển khai máy ảnh (Client)

```csharp
// Client (Nikon Z-Mount Camera)
public class NikonZMountCamera
{
    ...
}
```

### Triển khai lens Z (native)

```csharp
// Native Z-Mount Lens
public class NikonZMountLens : INikonZMount
{
    ...
}
```

### Triển khai lens F

```csharp
// Adaptee (Existing lens with F-mount)
public class NikonFMountLens
{
    ...
}
```

### Tạo Interface cho tất cả lens ngàm Z

```csharp
// Target Interface
public interface INikonZMount
{
    void AttachLens();
    void TakePicture();
}
```

### Tạo Adapter

```csharp
public class FMountToZMountAdapter : INikonZMount
{
    ...
}
```

### Sử dụng

```csharp
public class AdapterPatternDemo
{
    public static void Main(string[] args)
    {
        ...
    }
}
```

[OnlineGDB](https://onlinegdb.com/l5JIadcxG)

## Kết quả

```
Using native Z-Mount lens:
Mounting lens to Nikon Z-Mount camera.
...

Using F-Mount lens with adapter:
...
```

## Ưu và nhược điểm

### Ưu điểm

- Tính linh hoạt cao...
- Tuân thủ nguyên tắc Open/Closed...
- Tách biệt Client code với Interface...

### Nhược điểm

- Tăng độ phức tạp...
- Hiệu suất...
- Không đảm bảo mô phỏng 100%...
