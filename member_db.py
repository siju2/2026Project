import pymysql
import bcrypt

# DB 연결 설정
db_config = {
    'host': 'localhost',
    'user': 'member_user',  # root 대신 새로 만든 유저
    'password': '1234',      # 새로 설정한 비밀번호
    'db': 'member',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# --- [기존] 회원가입 함수 ---
def save_member(name, email, raw_password):
    hashed_pw = bcrypt.hashpw(raw_password.encode('utf-8'), bcrypt.gensalt())
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
            cursor.execute(sql, (name, email, hashed_pw.decode('utf-8')))
        conn.commit()
        return True
    except Exception as e:
        print(f"저장 오류: {e}")
        return False
    finally:
        conn.close()

# --- [신규] 로그인 확인 함수 ---
def login_check(email, raw_password):
    """
    입력받은 이메일로 DB에서 사용자를 찾고, 
    입력받은 비밀번호와 DB에 저장된 암호화 비밀번호가 일치하는지 확인합니다.
    """
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            # 1. DB에서 해당 이메일의 유저 정보 가져오기
            sql = "SELECT * FROM users WHERE email = %s"
            cursor.execute(sql, (email,))
            user = cursor.fetchone()

            if user:
                # 2. 비밀번호 대조 (입력한 비번 vs DB에 저장된 해시값)
                # bcrypt.checkpw는 알아서 해시를 풀어서 비교해줍니다.
                if bcrypt.checkpw(raw_password.encode('utf-8'), user['password'].encode('utf-8')):
                    return {"success": True, "name": user['name']} # 성공 시 이름 반환
                else:
                    return {"success": False, "message": "비밀번호가 틀렸습니다."}
            else:
                return {"success": False, "message": "존재하지 않는 이메일입니다."}
    finally:
        conn.close()

# --- 테스트 코드 ---
if __name__ == "__main__":
    print("--- 로그인 테스트 ---")
    input_email = input("이메일: ")
    input_pw = input("비밀번호: ")
    
    result = login_check(input_email, input_pw)
    if result["success"]:
        print(f"🎉 로그인 성공! 환영합니다, {result['name']}님.")
    else:
        print(f"❌ 로그인 실패: {result['message']}")