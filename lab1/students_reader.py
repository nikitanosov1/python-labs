from typing import Union, List, Dict
from collections import namedtuple
from datetime import date
import os.path
import json

LAB_WORK_SESSION_KEYS = ("presence", "lab_work_n", "lab_work_mark", "date")
STUDENT_KEYS = ("unique_id", "name", "surname", "group",
                "subgroup", "lab_works_sessions")


class LabWorkSession(namedtuple('LabWorkSession', 'presence, lab_date, lab_work_n, lab_work_mark')):

    @staticmethod
    def _validate_session(presence: bool, lab_work_n: int, lab_date: str, lab_work_mark: int):
        if not isinstance(presence, int):
            return False, 'bad_presence'
        if not isinstance(lab_work_n, int) or lab_work_n < 0:
            return False, 'bad_lab_work_n'
        if not isinstance(lab_work_mark, int) or lab_work_n < 0:
            return False, 'bad_lab_work_mark'
        if not isinstance(lab_date, str):
            return False, 'bad_lab_date'
        if presence == 0 and lab_work_mark != 0:
            return False, 'incorrect mark'
        return True, None

    def __new__(cls, presence: bool, lab_work_n: int, lab_work_mark: int, lab_date: str):
        is_validate, mes = LabWorkSession._validate_session(
            presence, lab_work_n, lab_date, lab_work_mark)
        if is_validate:
            t = tuple(map(int, lab_date.split(':')))
            lab_date = date(t[2], t[1], t[0])
            return super().__new__(cls, presence, lab_date, lab_work_n, lab_work_mark)
        raise ValueError(f'LabWorkSession :: {mes}')

    def __str__(self):
        return (f'{{\n'
                f'  "date": "{self.lab_date.day}:{self.lab_date.month}:{self.lab_date.year}", \n'
                f'  "presence": {self.presence}, \n'
                f'  "lab_work_n": {self.lab_work_n}, \n'
                f'  "lab_work_mark": {self.lab_work_mark}\n'
                f'}}')


class Student:
    __slots__ = ('_unique_id', '_name', '_surname',
                 '_group', '_subgroup', '_lab_work_sessions')

    def _build_student(self, name, surname, group, subgroup, unique_id):
        if not isinstance(unique_id, int) or unique_id < 0:
            return False, f'id = {unique_id} :: bad_id'
        if not isinstance(name, str) or len(name) == 0:
            return False, f'id = {unique_id} :: bad_name'
        if not isinstance(surname, str) or len(surname) == 0:
            return False, f'id = {unique_id} :: bad_surname'
        if not isinstance(group, int) or group <= 0:
            return False, f'id = {unique_id} :: bad_group'
        if not isinstance(subgroup, int) or subgroup <= 0:
            return False, f'id = {unique_id} :: bad_subgroup'
        self._name = name
        self._unique_id = unique_id
        self._surname = surname
        self._group = group
        self._subgroup = subgroup
        return True, None

    def append_lab_work_session(self, session: LabWorkSession):
        self._lab_work_sessions.append(session)

    def __init__(self, name: str, surname: str, group: int, subgroup: int, unique_id: int):
        is_correct, mes = self._build_student(
            name, surname, group, subgroup, unique_id)
        if not is_correct:
            raise ValueError(mes)
        self._lab_work_sessions: Union[List, None] = []

    def __str__(self):
        sep = ',\n'
        return (f'{{\n'
                f'"unique_id": {self._unique_id}, \n'
                f'"name": "{self._name}", \n'
                f'"surname": "{self._surname}", \n'
                f'"group": {self._group}, \n'
                f'"subgroup": {self._subgroup}, \n'
                f'"lab_works_sessions": [ \n'
                f'{sep.join(str(v) for v in self.lab_work_sessions)} ] \n'
                f'}}\n')

    def str_csv(self, session):
        return (f'{self.unique_id};"{self.name}";"{self.surname}";{self.group};{self.subgroup};'
                f'"{session.lab_date.day}:{session.lab_date.month}:{session.lab_date.year}";'
                f'{session.presence};{session.lab_work_n};{session.lab_work_mark}')

    @property
    def unique_id(self) -> int:
        return self._unique_id

    @property
    def group(self) -> int:
        return self._group

    @property
    def subgroup(self) -> int:
        return self._subgroup

    @property
    def name(self) -> str:
        return self._name

    @property
    def surname(self) -> str:
        return self._surname

    @name.setter
    def name(self, val: str) -> None:
        if not isinstance(val, str):
            raise ValueError()
        if len(val) == 0:
            raise ValueError()
        self._name = val

    @surname.setter
    def surname(self, val: str) -> None:
        if not isinstance(val, str):
            raise ValueError()
        if len(val) == 0:
            raise ValueError()
        self._surname = val

    @group.setter
    def group(self, val: int) -> None:
        if not isinstance(val, int):
            raise ValueError()
        if val < 1:
            raise ValueError()
        self._group = val

    @subgroup.setter
    def subgroup(self, val: int) -> None:
        if not isinstance(val, int):
            raise ValueError()
        if val < 1:
            raise ValueError()
        self._subgroup = val

    @property
    def lab_work_sessions(self):
        for session in self._lab_work_sessions:
            yield session


def _load_lab_work_session(json_node) -> LabWorkSession:
    """
        Создание из под-дерева json файла экземпляра класса LabWorkSession.
        hint: чтобы прочитать дату из формата строки, указанного в json используйте
        date(*tuple(map(int, json_node['date'].split(':'))))
    """
    for key in LAB_WORK_SESSION_KEYS:
        if key not in json_node:
            raise KeyError(
                f'load_lab_work_session:: key "{key}" not present in json_node')
    return LabWorkSession(json_node['presence'], json_node['lab_work_n'], json_node['lab_work_mark'], json_node['date'])


def _load_student(json_node) -> Student:
    """
        Создание из под-дерева json файла экземпляра класса Student.
        Если в процессе создания LabWorkSession у студента случается ошибка,
        создание самого студента ломаться не должно.
    """
    for key in STUDENT_KEYS:
        if key not in json_node:
            raise KeyError(
                f"load_student:: key \"{key}\" not present in json_node")
    student = Student(json_node['name'], json_node['surname'],
                      json_node['group'], json_node['subgroup'], json_node['unique_id'])
    for session in json_node['lab_works_sessions']:
        try:
            student.append_lab_work_session(_load_lab_work_session(session))
        except KeyError as error:
            print(error)
        except ValueError as error:
            print(error)
    return student


def load_students_json(file_path: str) -> Union[List[Student], None]:
    """
    Загрузка списка студентов из json файла.
    Ошибка создания экземпляра класса Student не должна приводить к поломке всего чтения.
    """
    assert isinstance(file_path, str)
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rt', encoding='UTF-8') as input_file:
        data = json.load(input_file)
        if 'students' not in data:
            return None
        data = data['students']
        students = []
        for student_json in data:
            try:
                students.append(_load_student(student_json))
            except KeyError as error:
                print(error)
            except ValueError as error:
                print(error)
    return students


def save_students_json(file_path: str, students: List[Student]):
    sep = ',\n'
    with open(file_path, 'w', encoding='UTF-8') as outfile:
        print(
            f'{{\n"students":[\n{sep.join(str(student) for student in students)}]\n}}', file=outfile, end='')


def load_students_csv(file_path: str) -> Union[List[Student], None]:
    """
    Загрузка списка студентов из json файла.
    Ошибка создания экземпляра класса Student не должна приводить к поломке всего чтения.
    """
    # csv header
    #     0    |   1  |   2   |   3  |    4    |  5  |    6    |        7       |       8     |
    # unique_id; name; surname; group; subgroup; date; presence; lab_work_number; lab_work_mark
    assert isinstance(file_path, str)  # Путь к файлу должен быть строкой
    if not os.path.exists(file_path):  # и, желательно, существовать...
        return None
    with open(file_path, 'rt', encoding='UTF-8') as input_file:
        students_raw: Dict[int, Student] = {}
        input_file.readline()
        for line in input_file:
            try:
                data = line.split(';')
                data[0] = int(data[0])
                if data[0] in students_raw:
                    students_raw[data[0]].append_lab_work_session(LabWorkSession(int(data[6]), int(data[7]), int(data[8]),
                                                                                 data[5].replace('"', '')))
                else:
                    student = Student(data[1].replace('"', ''), data[2].replace('"', ''), int(data[3]), int(data[4]),
                                      data[0])
                    student.append_lab_work_session(LabWorkSession(int(data[6]), int(data[7]), int(data[8]),
                                                                   data[5].replace('"', '')))
                    students_raw[data[0]] = student
            except Exception as ex:
                print(ex)
                continue
    return list(students_raw.values())


def save_students_csv(file_path: str, students: List[Student]):
    sep = '\n'
    data = (f'unique_id;name;surname;group;subgroup;date;presence;lab_work_number;lab_work_mark\n')
    with open(file_path, 'w', encoding='UTF-8') as outfile:
        print(data, file=outfile, end='')
        for student in students:
            for session in student.lab_work_sessions:
                print((f'{student.unique_id};"{student.name}";"{student.surname}";{student.group};{student.subgroup};'
                       f'"{session.lab_date.day}:{session.lab_date.month}:{session.lab_date.year}";'
                       f'{session.presence};{session.lab_work_n};{session.lab_work_mark}\n'), file=outfile, end='')


if __name__ == '__main__':
    # Проверка json
    students = load_students_json('lab1/students.json')
    save_students_json('lab1/students_saved.json', students)
    students_saved = load_students_json('lab1/students_saved.json')

    # Проверка csv
    students = load_students_csv('lab1/students.csv')
    save_students_csv('lab1/students_saved.csv', students)
    students_saved = load_students_csv('lab1/students_saved.csv')
