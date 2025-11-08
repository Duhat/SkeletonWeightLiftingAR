import socket

# Простая отправка одного сообщения
def quick_send():
    server_address = ('127.0.0.1', 65432)
    
    # Создаем UDP сокет
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Отправляем сообщение
    message = "Test message from Python"
    sock.sendto(message.encode('utf-8'), server_address)
    print(f"Sent: {message}")
    
    sock.close()

# Запуск
quick_send()