# Описание проблемы

Проблема при получении клиентом задания с boinc server'а (при этом задания на сервере имеются).

## Описание имеющихся файлов

*boinc_event_log - cache_full.txt* - Лог-файл Boinc manager'a. В конце файла указана причина, почему не загружаются задания с boinc server'a - кэш заполнен. Далее данная проблема была устранена.

*boinc_event_log - got_0_new_tasks.txt* - Лог-файл Boinc manager'a. В конце файла указано, что передано клиенту 0 заданий (хотя задания на сервере имеются).

*boinc_event_log - not_installed_Virtual_Box.png*- Лог-файл Boinc manager'a. В конце файла указана причина, почему не загружаются задания с boinc server'a - не установлен Virtual Box. Далее данная проблема была устранена.

![](https://github.com/xex238/Distributed_computing_2021/blob/main/boinc-docker-server%20problem/boinc_event_log%20-%20not_installed_Virtual_Box.png?raw=true)

*boinc_server_applications.png* - Отображение в boinc server'е списка поддерживаемых платформ для осуществления распределённых вычислений. Так, возможно производить вычисления на платформах:

- Windows

- Linux

- Mac OS

  ![](https://github.com/xex238/Distributed_computing_2021/blob/main/boinc-docker-server%20problem/boinc_server_applications.png?raw=tr)

*boinc_server_results.png* - Отображение в boinc server'e списка заданий проекта. На изображении показано, что на сервере имеется 2 ещё не решённых задания.

![](https://github.com/xex238/Distributed_computing_2021/blob/main/boinc-docker-server%20problem/boinc_server_results.png?raw=true)

*boinc_server_status.png* - Отображение в boinc server'e статус отдельных компонентов сервера. Все компоненты сервера запущены и работают стабильно.

![](https://github.com/xex238/Distributed_computing_2021/blob/main/boinc-docker-server%20problem/boinc_server_status.png?raw=true)

*scheduler.log* - Файл с логами sheduler'а с boinc server'a. Исходя из данного лога, можно сделать вывод, что sheduler по каким-то причинам не выдаёт задания клиентам.