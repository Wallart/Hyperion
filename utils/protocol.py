def frame_decode(frame):
    decoded = dict()
    frame_copy = frame.copy()
    while True:
        chunk_header = frame_copy[:3].decode('utf-8')
        chunk_size = int.from_bytes(frame_copy[3:7], 'big')
        chunk_content = frame_copy[7:7+chunk_size]
        if len(chunk_content) < chunk_size:
            return None

        if chunk_header == 'PCM':
            decoded[chunk_header] = chunk_content
        elif chunk_header == 'ANS':
            decoded['IDX'] = chunk_content[0]
            decoded[chunk_header] = chunk_content[1:].decode('utf-8')
        else:
            decoded[chunk_header] = chunk_content.decode('utf-8')

        frame_copy = frame_copy[7+chunk_size:]
        if chunk_header == 'PCM':
            return decoded, frame_copy


def frame_encode(idx, request, answer, pcm):
    # beware of accents, they are using 2 bytes. Byte string might be longer than str
    answer = int.to_bytes(idx, 1, 'big') + bytes(answer, 'utf-8')
    request = bytes(request, 'utf-8')

    req_len = len(request)
    ans_len = len(answer)
    pcm_len = len(pcm) * 2  # because each value is coded on 2 bytes (16 bits)

    frame = bytes('REQ', 'utf-8')
    frame += req_len.to_bytes(4, 'big')  # big = read bytes from left to right
    frame += request

    frame += bytes('ANS', 'utf-8')
    frame += ans_len.to_bytes(4, 'big')
    frame += answer

    frame += bytes('PCM', 'utf-8')
    frame += pcm_len.to_bytes(4, 'big')
    frame += pcm.tobytes()

    return frame
