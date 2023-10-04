import csv
from typing import Union, List, Dict
from collections import namedtuple
from datetime import date
import os.path
import json


LAB_WORK_SESSION_KEYS = ("presence", "lab_work_n", "lab_work_mark", "date")
STUDENT_KEYS = ("unique_id", "name", "surname", "group", "subgroup", "lab_works_sessions")


class LabWorkSession(namedtuple('LabWorkSession', 'presence, lab_work_number, lab_work_mark, lab_work_date')):
    """
    Информация о лабораторном занятии, которое могло или не могло быть посещено студентом
    """
    def __new__(cls, presence: bool, lab_work_number: int, lab_work_mark: int, lab_work_date: date):
        """
            param: presence: присутствие студента на л.р.(bool)
            param: lab_work_number: номер л.р.(int)
            param: lab_work_mark: оценка за л.р.(int)
            param: lab_work_date: дата л.р.(date)
        """
        if not LabWorkSession._validate_session(presence, lab_work_number, lab_work_mark, lab_work_date):
            raise ValueError(f"LabWorkSession ::"
                    f"incorrect args :\n"
                    f"presence       : {presence},\n"
                    f"lab_work_number: {lab_work_number},\n"
                    f"lab_work_mark  : {lab_work_mark},\n"
                    f"lab_work_date  : {lab_work_date}")
        
        obj = super().__new__(cls, presence, lab_work_number, lab_work_mark, lab_work_date)
        return obj

    @staticmethod
    def _validate_session(presence: bool, lab_work_number: int, lab_work_mark: int, lab_work_date: date) -> bool:
        """
            param: presence: присутствие студента на л.р.(bool)
            param: lab_work_number: номер л.р.(int)
            param: lab_work_mark: оценка за л.р.(int)
            param: lab_work_date: дата л.р.(date)
        """
        if not isinstance(presence, bool):
            return False
        if not isinstance(lab_work_number, int) or lab_work_number < 0:
            return False
        if not isinstance(lab_work_mark, int) or lab_work_mark < 0 or lab_work_mark > 5:
            return False
        if not isinstance(lab_work_date, date):
            return False
        return True

    def __str__(self) -> str:
        """
            Строковое представление LabWorkSession
            Пример:
            {
                    "presence":      1,
                    "lab_work_n":    4,
                    "lab_work_mark": 3,
                    "date":          "15:12:23"
            }
        """

        data = {
            'presence': self.presence,
            'lab_work_n': self.lab_work_number,
            'lab_work_mark': self.lab_work_mark,
            'date': self.lab_work_date.strftime("%d:%m:%Y"),
        }
        return json.dumps(data, indent=4, ensure_ascii=False)

    def convert_to_dict_with_custom_field_names(self, presence_name, lab_work_number_name, lab_work_mark_name, lab_work_date_name) -> dict:
        return {
            presence_name: self.presence,
            lab_work_number_name: self.lab_work_number,
            lab_work_mark_name: self.lab_work_mark,
            lab_work_date_name: self.lab_work_date.strftime("%d:%m:%Y")
        }
        
class Student:
    __slots__ = ('_unique_id', '_name', '_surname', '_group', '_subgroup', '_lab_work_sessions')

    def __init__(self, unique_id: int, name: str, surname: str, group: int, subgroup: int):
        """
            param: unique_id: уникальный идентификатор студента (int)
            param: name: имя студента (str)
            param: surname: фамилия студента (str)
            param: group: номер группы в которой студент обучается (int)
            param: subgroup: номер подгруппы (int)
        """
        if not self._build_student(unique_id, name, surname, group, subgroup):
            raise ValueError(f"Student ::"
                             f"incorrect args :\n"
                             f"unique_id : {unique_id},\n"
                             f"name      : {name},\n"
                             f"surname   : {surname},\n"
                             f"group     : {group},\n"
                             f"subgroup  : {subgroup}")
    
        self._unique_id = unique_id
        self._name = name
        self._surname = surname
        self._group = group
        self._subgroup = subgroup
        self._lab_work_sessions: List[LabWorkSession] = []

    def _build_student(self, unique_id: int, name: str, surname: str, group: int, subgroup: int) -> bool:
        """
            param: unique_id: уникальный идентификатор студента (int)
            param: name: имя студента (str)
            param: surname: фамилия студента (str)
            param: group: номер группы в которой студент обучается (int)
            param: subgroup: номер подгруппы (int)
        """
        if not isinstance(unique_id, int) or unique_id < 0:
            return False
        if not isinstance(name, str) or not name:
            return False
        if not isinstance(surname, str) or not surname:
            return False
        if not isinstance(group, int) or group <= 0:
            return False
        if not isinstance(subgroup, int) or subgroup <= 0:
            return False
        return True

    def __str__(self) -> str:
        """
        Строковое представление Student
        Пример:
        {
                "unique_id":          26,
                "name":               "Щукарев",
                "surname":            "Даниил",
                "group":              6408,
                "subgroup":           2,
                "lab_works_sessions": [
                    {
                        "presence":      1,
                        "lab_work_n":    1,
                        "lab_work_mark": 4,
                        "date":          "15:9:23"
                    },
                    {
                        "presence":      1,
                        "lab_work_n":    2,
                        "lab_work_mark": 4,
                        "date":          "15:10:23"
                    },
                    {
                        "presence":      1,
                        "lab_work_n":    3,
                        "lab_work_mark": 4,
                        "date":          "15:11:23"
                    },
                    {
                        "presence":      1,
                        "lab_work_n":    4,
                        "lab_work_mark": 3,
                        "date":          "15:12:23"
                    }]
        }
        """
        data = {
            'unique_id': self._unique_id,
            'name': self._name,
            'surname': self._surname,
            'group': self._group,
            'subgroup': self._subgroup,
            'lab_works_sessions':  [lab_work_session.convert_to_dict_with_custom_field_names("presence", "lab_work_n", "lab_work_mark","date") for lab_work_session in self._lab_work_sessions],
        }
        return json.dumps(data, indent=4, ensure_ascii=False)

    @property
    def unique_id(self) -> int:
        """
        Метод доступа для unique_id
        """
        return self._unique_id

    @property
    def group(self) -> int:
        """
        Метод доступа для номера группы
        """
        return self.group

    @property
    def subgroup(self) -> int:
        """
        Метод доступа для номера подгруппы
        """
        return self.subgroup

    @property
    def name(self) -> str:
        """
        Метод доступа для имени студента
        """
        return self.name

    @property
    def surname(self) -> str:
        """
        Метод доступа для фамилии студента
        """
        return self.surname

    @name.setter
    def name(self, val: str) -> None:
        """
        Метод для изменения значения имени студента
        """
        self.name = val

    @surname.setter
    def surname(self, val: str) -> None:
        """
        Метод для изменения значения фамилии студента
        """
        self.surname = val

    @property
    def lab_works_sessions(self):
        """
        Метод доступа для списка лабораторных работ, которые студент посетил или не посетил
        """
        return self._lab_work_sessions

    def append_lab_work_session(self, session: LabWorkSession):
        """
        Метод для регистрации нового лабораторного занятия
        """
        self._lab_work_sessions.append(session)
        


def _load_lab_work_session(json_node) -> LabWorkSession:
    """
        Создание из под-дерева json файла экземпляра класса LabWorkSession.
        hint: чтобы прочитать дату из формата строки, указанного в json используйте
        date(*tuple(map(int, json_node['date'].split(':'))))
    """
    for key in LAB_WORK_SESSION_KEYS:
        if key not in json_node:
            raise KeyError()

    preDate = json_node["date"].split(":")[::-1]
    if len(preDate[0]) < 4:
        preDate[0] = "20" + preDate[0]
    return LabWorkSession(True if int(json_node['presence']) == 1 else False,
                          int(json_node["lab_work_n"]),
                          int(json_node["lab_work_mark"]),
                          date(*map(int, preDate)))


def _load_student(json_node) -> Student:
    """
        Создание из под-дерева json файла экземпляра класса Student.
        Если в процессе создания LabWorkSession у студента случается ошибка,
        создание самого студента ломаться не должно.
    """
    for key in STUDENT_KEYS:
        if key not in json_node:
            raise KeyError(f"load_student:: key \"{key}\" not present in json_node")

    student = Student(json_node['unique_id'], json_node['name'], json_node['surname'], int(json_node['group']), int(json_node['subgroup']))

    for session_data in json_node['lab_works_sessions']:
        try:
            session = _load_lab_work_session(session_data)
            student.append_lab_work_session(session)
        except ValueError as er:
            print(er)
            continue
        except KeyError as er:
            print(er)
            continue
        except Exception as er:
            print(er)
            continue
    return student


# csv header
#     0    |   1  |   2   |   3  |    4    |  5  |    6    |        7       |       8     |
# unique_id; name; surname; group; subgroup; date; presence; lab_work_number; lab_work_mark
UNIQUE_ID = 0
STUD_NAME = 1
STUD_SURNAME = 2
STUD_GROUP = 3
STUD_SUBGROUP = 4
LAB_WORK_DATE = 5
STUD_PRESENCE = 6
LAB_WORK_NUMBER = 7
LAB_WORK_MARK = 8


def load_students_csv(file_path: str) -> Union[List[Student], None]:
     assert isinstance(file_path, str)
     if not os.path.exists(file_path):
         return None

     students = []

     with open(file_path, 'r', encoding='utf-8') as file:
         csv_reader = csv.reader(file, delimiter=';')
         header = next(csv_reader)  # Пропускаем заголовок

         for row in csv_reader:
             unique_id = int(row[UNIQUE_ID])
             name = row[STUD_NAME]
             surname = row[STUD_SURNAME]
             group = int(row[STUD_GROUP])
             subgroup = int(row[STUD_SUBGROUP])

             student = Student(unique_id, name, surname, group, subgroup)

             # Добавляем информацию о лабораторных работах
             lab_work_preDate = row[LAB_WORK_DATE].split(':')[::-1]
             if len(lab_work_preDate[0]) < 4:
                 lab_work_preDate[0] = "20" + lab_work_preDate[0]
             lab_work_date = date(*map(int, lab_work_preDate))
             presence = bool(int(row[STUD_PRESENCE]))
             lab_work_number = int(row[LAB_WORK_NUMBER])
             lab_work_mark = int(row[LAB_WORK_MARK])

             lab_work_session = LabWorkSession(presence, lab_work_number, lab_work_mark, lab_work_date)
             student.append_lab_work_session(lab_work_session)

             students.append(student)

     return students


def load_students_json(file_path: str) -> Union[List[Student], None]:
    """
    Загрузка списка студентов из json файла.
    Ошибка создания экземпляра класса Student не должна приводить к поломке всего чтения.
    """
    students = []
    with open(file_path, encoding="utf8") as f:
        file_json = json.load(f)
        for student_json in file_json['students']:
            student = _load_student(student_json)
            for lab_work_session_json in student_json['lab_works_sessions']:
                lab_word_session = _load_lab_work_session(lab_work_session_json)
                student.append_lab_work_session(lab_word_session)
            students.append(student)
    return students
    
def save_students_json(file_path: str, students: List[Student]):
    """
    Запись списка студентов в json файл
    """
    with open(file_path, 'w', encoding="utf8") as file:
        sep = ',\n'
        print(f"{{\n\t\"students\":[\n{sep.join(str(v) for v in students)}]\n}}", file=file)

def save_students_csv(file_path: str, students: List[Student]):
    """
    Запись списка студентов в csv файл
    """
    assert isinstance(file_path, str)

    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        csv_writer = csv.writer(file, delimiter=';')

        # Записываем заголовок
        header = [
            "unique_id", "name", "surname", "group", "subgroup",
            "date", "presence", "lab_work_number", "lab_work_mark"
        ]
        csv_writer.writerow(header)

        for student in students:
            for session in student._lab_work_sessions:
                row = [
                    student._unique_id, student._name, student._surname, student._group, student._subgroup,
                    session.lab_work_date.strftime("%d:%m:%Y"), int(session.presence),
                    session.lab_work_number, session.lab_work_mark
                ]
                csv_writer.writerow(row)

if __name__ == '__main__':
    # Задание на проверку json читалки:
    # 1. прочитать файл "students.json"
    # 2. сохранить прочитанный файл в "saved_students.json"
    # 3. прочитать файл "saved_students.json"
    # Задание на проверку csv читалки:
    # 1.-3. аналогично

    # Проверка json
    students = load_students_json('lab1/students.json')
    save_students_json('lab1/saved_students.json', students)
    students = load_students_json('lab1/saved_students.json')
    
    # Проверка csv
    # students = load_students_csv('lab1/students.csv')
    # save_students_csv('lab1/saved_students.csv', students)
    # students = load_students_csv('lab1/saved_students.csv')