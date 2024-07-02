using System;

namespace MvcMovie.Models
{
    public class Event
    {
        private string _name;
        private string _type;
        private DateTime _date;

        public string Name
        {
            get
            {
                return this._name;
            }
            set
            {
                this._name = value;
            }
        }

        public string Type
        {
            get
            {
                return this._type;
            }
            set
            {
                this._type = value;
            }
        }

        public DateTime Date
        {
            get
            {
                return this._date;
            }
            set
            {
                this._date = value;
            }
        }

        public Event(string _name, string _type, DateTime _date)
        {
            this._name = _name;
            this._type = _type;
            this._date = _date;
        }
    }
}
