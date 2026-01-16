def count_message(msg, count=0):
    count += 1
    print(f"Message: {msg}, Count: {count}")
    return count

count_message("Hello")