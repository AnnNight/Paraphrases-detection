import gradio as gr
import algorithms
import shutil
import os as os

output_par = None
output_mgen = None


def instruction_phrases(code):
    mes = ""
    if code == "start":
        mes = "Для начала работы в сервисе пройдите процедуру Авторизации или Регистрации на одноименной вкладке"
    elif code == "error_existing_login":
        mes = "Уже существует такой логин. Если это ваша учетная запись, то авторизуйтесь на вкладке \"Авторизация\""
    elif code == "success_registration":
        mes = "Регистрация прошла успешно!\nЧтобы начать проверку, перейдите на вкладку \"Загрузка файлов\".\n" \
              "Выберите интересующий вас тип и вид проверки, Загрузите соответствующие файлы," \
              "нажмите \"Начать проверку\" "
    elif code == "registration_error":
        mes = "Произошла ошибка при регистрации. Попробуйте еще раз"
    elif code == "error_logged_in":
        mes = "Вы уже авторизованы в системе. Не удалось зарегистрироваться.\nДля начала проверки проследуйте на вкладку" \
              "\"Загрузка файлов\""
    elif code == "success_log_in":
        mes = "Вы успешно авторизовались!\n1)Чтобы начать проверку, перейдите на вкладку \"Загрузка файлов\".\n" \
              "2)Выберите интересующий вас тип и вид проверки\n3)Загрузите соответствующие файлы, нажмите " \
              "\"Начать проверку\"\n4)Дождитесь появления сообщения об успешном завершении проверки" \
              "\n5)Перейдите на страницу результатов для изучения результатов проверки" \
              "\n6)Если в поле \"Файл для проверки\" ничего нет, либо не то, что вы загружали, или возникает ошибка - нажмите кнопку \"Обновить список\"" \
              "\n7) После завершения изучения результатов текущей проверки нажмите кнопку \"Завершить проверку\"." \
              "Без этого шага мы не можем гарантировать, что ваша следующая проверка будет успешной"
    elif code == "error_log_in":
        mes = "Вам не удалось авторизоваться.\nПопробуйте еще раз, либо перейдите на вкладку \"Зарегистрироваться\""
    return mes


def registration(username, password, phoneNumber, em, user_id):
    if user_id is not None and user_id != "" and int(user_id) > -1:
        return user_id, instruction_phrases("error_logged_in")
    ui = algorithms.find_login(username)
    if ui > -1:
        return str(
            -1), instruction_phrases("error_existing_login")
    else:
        ui = algorithms.create_user(username, password, phoneNumber, em)
        if ui > -1:
            mes = instruction_phrases("success_registration")
        else:
            mes = instruction_phrases("registration_error")
    return ui, mes


def upload_and_rename_file(file, upload_dir):
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    if file is not None:
        new_file_path = os.path.join(upload_dir)
        if type(file) is list:
            file_names = []
            for each in file:
                shutil.copy(each.name, new_file_path)
                n = each.name.split("\\")
                name_ = n[-1]
                file_names.append(name_)
            for zfile in os.listdir(path=new_file_path):
                if zfile not in file_names:
                    os.remove(new_file_path + "/" + zfile)
        else:
            for zfile in os.listdir(path=new_file_path):
                if zfile != file.name:
                    os.remove(new_file_path + "/" + zfile)
            shutil.copy(file.name, new_file_path)


def update_path(user_id, UploadDirFile, UploadDirFiles):
    UploadDirFile = "./data/" + str(user_id) + "/" + str('0') + "/susp"
    UploadDirFiles = "./data/" + str(user_id) + "/" + str('0') + "/src"
    return UploadDirFile, UploadDirFiles


with (gr.Blocks() as service):
    user_id = gr.Textbox(visible=False)
    output_p = gr.Textbox(label="Вывод", visible=False)
    instruction = gr.Textbox(label="Инструкция", visible=True,
                             value=instruction_phrases("start"))
    with gr.Tab(label="Авторизация") as auth:
        md = gr.Markdown("Введите логин и пароль, чтобы авторизоваться")
        mdError = gr.Markdown("Логин и пароль не соответствуют", visible=False)
        login = gr.Textbox(label="Логин")
        password = gr.Textbox(label="Пароль", type="password")
        btnEnter = gr.Button("Войти")


        def authentification(username_, password_):
            user_id_ = str(algorithms.find_user(username_, password_))
            if user_id_ != str(-1):
                mes = instruction_phrases("success_log_in")
                return str(user_id_), mes
            return str(-1), instruction_phrases("error_log_in")


        btnEnter.click(authentification, inputs=[login, password],
                       outputs=[user_id, instruction])
    with gr.Tab(label="Регистрация", visible=True) as register:
        mdR = gr.Markdown("Введите данные:")
        login_r = gr.Textbox(label="Логин")
        password_r = gr.Textbox(label="Пароль", type="password")
        phoneNumber = gr.Textbox(label="Номер телефона")
        email = gr.Textbox(label="Email")
        btnReg = gr.Button("Зарегистрироваться")
        btnReg.click(registration, inputs=[login_r, password_r, phoneNumber, email, user_id],
                     outputs=[user_id, instruction])

    with gr.Tab(label="Загрузка данных", visible=True) as SettingsPage:
        def update_dropdown(selected_radio):
            if selected_radio == 'Перефразирование':
                return gr.update(visible=True, choices=["Проверка файла в общей базе",
                                                        "Проверка одного файла относительно загруженных",
                                                        "Проверка всех загруженных файлов относительно всех"],
                                 value=None), gr.update(visible=False, value=None)
            elif selected_radio == 'Машинно-сгенерированный текст':
                return gr.update(visible=False, value=None), gr.update(visible=True, choices=["Проверка одного файла",
                                                                                  "Проверка нескольких файлов"],
                                                           value=None)
            else:
                return gr.update(visible=False), gr.update(visible=False)


        typeOfCheck = gr.Radio(["Перефразирование", "Машинно-сгенерированный текст"], label='Тип проверки:')
        typeOfCheckP = gr.Dropdown(["Проверка файла в общей базе", "Проверка одного файла относительно загруженных",
                                    "Проверка всех загруженных файлов относительно всех"], label="Вид проверки:",
                                   interactive=True)
        typeOfCheckMG = gr.Dropdown(["Проверка одного файла", "Проверка нескольких файлов"],
                                    label="Вид проверки:", visible=False, interactive=True)

        with gr.Row():
            files_input = gr.File(file_count="multiple", label="Файлы для проверки", interactive=True,
                                  file_types=['.txt'])

            file_ = gr.File(label="Проверяемый файл", interactive=True, file_types=['.txt'])
            UploadDirFile = gr.Textbox(value="./data/" + str(user_id.value) + "/" + str('0') + "/susp", visible=False)
            UploadDirFiles = gr.Textbox("./data/" + str(user_id.value) + "/" + str('0') + "/src",
                                        visible=False)  # check_id
            user_id.change(fn=update_path, inputs=[user_id, UploadDirFile, UploadDirFiles],
                           outputs=[UploadDirFile, UploadDirFiles])


            def handle_upload(file, directoryName):
                upload_and_rename_file(file, directoryName)


            file_.change(fn=handle_upload, inputs=[file_, UploadDirFile])
            files_input.change(fn=handle_upload, inputs=[files_input, UploadDirFiles])

        filesToBase = gr.Checkbox(label="Добавить загруженные файлы в общую базу")

        StartCheck = gr.Button("Начать проверку")
        resultText = gr.Textbox(label="Вывод", value="Начните проверку после загрузки файлов", visible=True,
                                placeholder="Output text")


    def check(typeOfCheckP, typeOfCheckMG, typeOfCheck, user_id, output, filesToBase):
        global output_par
        global output_mgen
        if user_id is None or user_id == "-1" or user_id == "":
            return "Начните проверку после загрузки файлов", instruction_phrases("start")
        if typeOfCheck == "Перефразирование":
            [output_par, output] = algorithms.check_paraphrased_text(typeOfCheckP, user_id, filesToBase)
        else:
            [output_mgen, output] = algorithms.check_generated_text(typeOfCheckMG, filesToBase, user_id)
        return output, instruction_phrases("success_log_in")


    StartCheck.click(check, inputs=[typeOfCheckP, typeOfCheckMG, typeOfCheck, user_id, resultText, filesToBase],
                     outputs=[resultText, instruction])

    with gr.Tab(label="Результат проверки", visible=True) as CheckResult:
        with gr.Column():
            with gr.Row():
                Chosen_file = gr.Textbox(label="Файл для проверки:", interactive=False, visible=False)
                Choose_file = gr.Dropdown([], label="Файл для проверки:", interactive=True, visible=True)
                btnUpdateList = gr.Button("Обновить список", interactive=True, visible=True)
            btnShowResult = gr.Button(value="Показать результат", visible=True)
            btnEndCheck = gr.Button(value="Завершить проверку", visible=True)


        def update_file_dropdown(typeOfCheckP, typeOfCheckMG, user_id, resultText, check_id='0'):
            print(typeOfCheckP, " ", typeOfCheckMG, " ", user_id)
            path1 = "./data/" + str(user_id) + "/" + str(check_id) + "/susp"
            path2 = "./data/" + str(user_id) + "/" + str(check_id) + "/src"
            if not os.path.exists(path1):
                os.makedirs(path1)
            if not os.path.exists(path2):
                os.makedirs(path2)
            files1 = os.listdir(path1)
            files2 = os.listdir(path2)
            if resultText != "Проверка завершена, перейдите на следующую вкладку, чтобы узнать результат":
                upload_and_rename_file([], path1)
                upload_and_rename_file([], path2)
            if typeOfCheckP == "Проверка одного файла относительно загруженных":
                return gr.update(visible=True, value=files1[0] if len(files1) != 0 else "") \
                    , gr.update(visible=False, value=None), gr.update(visible=True, value=None), gr.update(
                    visible=True, value=None), \
                    gr.update(visible=True, value=None)

            if typeOfCheckP == "Проверка файла в общей базе":
                return gr.update(visible=True, value=files1[0] if len(files1) != 0 else ""), \
                    gr.update(visible=False, value=None), gr.update(visible=True, value=None), gr.update(
                    visible=False, value=None), \
                    gr.update(visible=False, value=None)

            elif typeOfCheckP == "Проверка всех загруженных файлов относительно всех":
                return gr.update(visible=False, value=None), gr.update(visible=True, choices=files2), \
                    gr.update(visible=False, value=None), gr.update(visible=True, value=None), \
                    gr.update(visible=True, value=None)
            if typeOfCheckMG == "Проверка одного файла":
                return gr.update(visible=True, value=files1[0] if len(files1) != 0 else ""), \
                    gr.update(visible=False, value=None), gr.update(visible=True, value=None), \
                    gr.update(visible=False, value=None), gr.update(visible=True, value=None)
            elif typeOfCheckMG == "Проверка нескольких файлов":
                return gr.update(visible=False, value=None), gr.update(visible=True, choices=files2), \
                    gr.update(visible=False, value=None), gr.update(visible=True, value=None), \
                    gr.update(visible=True, value=None)
            else:
                return gr.update(visible=False, value=None), gr.update(visible=False, choices=files2), \
                    gr.update(visible=False, value=None), gr.update(visible=False, value=None), \
                    gr.update(visible=True, value=None)


        typeOfCheck.change(fn=update_dropdown, inputs=typeOfCheck,
                           outputs=[typeOfCheckP, typeOfCheckMG])
        typeOfCheckP.change(update_file_dropdown, inputs=[typeOfCheckP, typeOfCheckMG, user_id, resultText],
                            outputs=[Chosen_file, Choose_file, file_, files_input, filesToBase])
        typeOfCheckMG.change(update_file_dropdown, inputs=[typeOfCheckP, typeOfCheckMG, user_id, resultText],
                             outputs=[Chosen_file, Choose_file, file_, files_input, filesToBase])


        def highlight_text(output_p, Choose_file, Chosen_file):
            highlight = {"entities": []}
            text = ""
            start = 0
            if output_p is None:
                return {"text": "", "entities": []}
            results = []
            if Choose_file is not None and Choose_file != "":
                results = output_p[Choose_file]
            elif Chosen_file is not None and Chosen_file != "":
                results = output_p[Chosen_file]
            for res in results:
                text += res[0]
                end = start + len(res[0])
                highlight["entities"].append(
                    {
                        "entity": res[1],
                        "start": start,
                        "end": end,
                    }
                )
                start = end
            highlight["text"] = text
            return highlight


        def highlight_text_mg(output_mg, Choose_file, Chosen_file):
            highlight = {"entities": []}
            text = ""
            start = 0
            if output_mg is None:
                return {"text": "", "entities": []}
            results = []
            if Choose_file is not None and Choose_file != "":
                results = output_mg[Choose_file]
            elif Chosen_file is not None and Chosen_file != "":
                results = output_mg[Chosen_file]
            for res in results:
                text += res[0]
                end = start + len(res[0])
                highlight["entities"].append(
                    {
                        "entity": res[1],
                        "start": start,
                        "end": end,
                    }
                )
                start = end
            highlight["text"] = text
            return highlight


        path = "./data/" + str(user_id.value) + "/" + str('0') + "/src"
        if not os.path.exists(path):
            os.makedirs(path)
        files = os.listdir(path)
        colors = ["red"] * len(files)
        color_map = dict(zip(files, colors))
        text1_p = gr.HighlightedText(highlight_text(output_par, Choose_file, Chosen_file),
                                     label="Результат проверки:")  # , color_map=color_map)


        def update_result_2(Choose_file, Chosen_file, typeOfCheck, user_id, typeOfCheckP, typeOfCheckMG):
            if Choose_file == "" and Chosen_file == "":
                return Chosen_file, Choose_file, None
            if typeOfCheck == "Перефразирование":
                highlight = highlight_text(output_par, Choose_file, Chosen_file)
            else:
                highlight = highlight_text_mg(output_mgen, Choose_file, Chosen_file)
            return Chosen_file, Choose_file, highlight


        Choose_file.change(update_result_2, inputs=[Choose_file, Chosen_file,
                                                    typeOfCheck, user_id, typeOfCheckP, typeOfCheckMG],
                           outputs=[Chosen_file, Choose_file, text1_p])
        Chosen_file.change(update_result_2, inputs=[Choose_file, Chosen_file,
                                                    typeOfCheck, user_id, typeOfCheckP, typeOfCheckMG],
                           outputs=[Chosen_file, Choose_file, text1_p])
        btnUpdateList.click(update_file_dropdown, inputs=[typeOfCheckP, typeOfCheckMG, user_id, resultText],
                            outputs=[Chosen_file, Choose_file, file_, files_input, filesToBase])
        btnShowResult.click(update_result_2, inputs=[Choose_file, Chosen_file,
                                                     typeOfCheck, user_id, typeOfCheckP, typeOfCheckMG],
                            outputs=[Chosen_file, Choose_file, text1_p])


        def clear_all(user_id):
            directoryName1 = "./data/" + str(user_id) + "/0/susp"
            directoryName2 = "./data/" + str(user_id) + "/0/src"
            upload_and_rename_file([], directoryName1)
            upload_and_rename_file([], directoryName2)
            Chosen_file = ""
            Choose_file = ""
            return Chosen_file, Choose_file


        btnEndCheck.click(clear_all, inputs=[user_id], outputs=[Chosen_file, Choose_file])

service.launch()
